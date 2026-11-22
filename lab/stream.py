"""Capture the residual stream from a model too large to hold in memory.

The lab's normal path (model.py) loads the whole BF16 model into unified memory.
That works up to ~70 GB (Qwen3.6). Nemotron 3.5 Super VL is ~240 GB in BF16 and
does not fit on one 128 GB box. But measuring refusal directions only needs the
last-token hidden state at every layer boundary, which needs only one layer's
weights resident at a time.

StreamLab builds the language model on the meta device (weights are pointers to
nothing) and runs the forward "layer-outer": hold every prompt's activations at
once (a few GB), then for each of the 88 layers materialise that one layer's
weights from the memory-mapped BF16 shards, push every prompt through it, record
the hidden state, and free the layer before the next. Each layer's weights are
read from disk exactly once for the whole prompt set.

Only the language model is built: the vision tower and the MTP head are not
loaded (text prompts, no draft path). Checkpoint keys are
`language_model.backbone.*`; the standalone NemotronHModel wants `*`, so the
prefix is stripped when a tensor is loaded.

Mamba layers carry state along the sequence, so batches are RIGHT-padded and the
hidden state is gathered at each row's true last token. Attention runs eager so
the mask is a plain additive tensor built here, not a backend-specific object.
Only prompt-end capture is implemented -- the contrast every shipped build used.
"""
import inspect
import json
import os

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .model import DEFAULT_SYSTEM, guard, memory_report


def _desanitize(x):
    """NVIDIA encodes non-finite floats in config.json as {"__float__": "Infinity"}."""
    if isinstance(x, dict) and set(x) == {"__float__"}:
        return float(x["__float__"])
    if isinstance(x, dict):
        return {k: _desanitize(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_desanitize(v) for v in x]
    return x


def _strip(ckpt_key):
    """Checkpoint key -> standalone NemotronHModel key, or None to skip (vision/mtp)."""
    p = "language_model.backbone."
    return ckpt_key[len(p):] if ckpt_key.startswith(p) else None


class StreamLab:
    def __init__(self, model_dir, device="cuda", dtype=torch.bfloat16):
        self.model_dir = model_dir
        self.device = device
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Right padding: mamba's scan would otherwise fold padding into the state
        # of every real token. The true last index is gathered explicitly.
        self.tokenizer.padding_side = "right"

        cfg = _desanitize(json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8")))
        llm = cfg["llm_config"]
        llm["use_mamba_kernels"] = False          # no mamba_ssm/causal_conv1d kernels in the image
        if isinstance(llm.get("time_step_limit"), list):
            llm["time_step_limit"] = tuple(float(v) for v in llm["time_step_limit"])
        mt = llm.pop("model_type")
        self.config = AutoConfig.for_model(mt, **llm)
        self.config._attn_implementation = "eager"   # plain additive masks we can build by hand
        with torch.device("meta"):
            full = AutoModelForCausalLM.from_config(self.config)
        # Only the inner backbone is needed (hidden states, not logits).
        self.backbone = full.model
        self.backbone.eval()
        self.n_layers = self.config.num_hidden_layers

        # Where each needed tensor lives, by its stripped (standalone) name.
        wm = json.load(open(os.path.join(model_dir, "model.safetensors.index.json"), encoding="utf-8"))["weight_map"]
        self.where = {}
        for ckpt_key, shard in wm.items():
            s = _strip(ckpt_key)
            if s is not None:
                self.where[s] = (os.path.join(model_dir, shard), ckpt_key)

    @property
    def hidden_size(self):
        return self.config.hidden_size

    def __len__(self):
        return self.n_layers

    # --- prompts (same rendering as Lab) -----------------------------------------

    def render(self, prompt, thinking=False, system=DEFAULT_SYSTEM, messages=None):
        if messages is None:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=thinking
        )

    def tokenize(self, texts):
        return self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)

    # --- weight streaming --------------------------------------------------------

    def _load_module(self, module, prefix):
        """Materialise `module`'s parameters/buffers from the shards, on GPU.

        prefix is the module's name in the standalone model, e.g. "layers.3.".
        Most tensors map one-to-one to a checkpoint key (after stripping). The
        exception is a MoE layer's routed experts: the model holds them stacked
        as experts.up_proj [E, out, in] / experts.down_proj, while the checkpoint
        stores one matrix per expert (experts.<e>.up_proj.weight), so those are
        assembled by stacking in expert order. Reads are grouped by shard file so
        each file is opened once.
        """
        from collections import defaultdict

        direct, stacks = {}, {}       # local -> (file, key) ; local -> [(file, key), ...]
        for name, _ in module.named_parameters(recurse=True):
            full = prefix + name
            if full in self.where:
                direct[name] = self.where[full]
            elif name.endswith("experts.up_proj") or name.endswith("experts.down_proj"):
                leaf = name.rsplit(".", 1)[1]
                base = full[: -len("." + leaf)]   # ...mixer.experts
                stacks[name] = [self.where[f"{base}.{e}.{leaf}.weight"]
                                for e in range(self.config.n_routed_experts)]
            else:
                raise KeyError(f"no checkpoint tensor for {full}")
        for name, _ in module.named_buffers(recurse=True):
            if prefix + name in self.where:       # persisted buffers only
                direct[name] = self.where[prefix + name]

        byfile = defaultdict(list)                # file -> [(key, target), ...]
        for local, (file, key) in direct.items():
            byfile[file].append((key, ("d", local)))
        parts = {local: [None] * len(lst) for local, lst in stacks.items()}
        for local, lst in stacks.items():
            for e, (file, key) in enumerate(lst):
                byfile[file].append((key, ("s", local, e)))

        loaded = {}
        for file, items in byfile.items():
            with safe_open(file, framework="pt") as f:
                for key, tgt in items:
                    t = f.get_tensor(key).to(self.device, self.dtype)
                    if tgt[0] == "d":
                        loaded[tgt[1]] = t
                    else:
                        parts[tgt[1]][tgt[2]] = t
        for local, ps in parts.items():
            loaded[local] = torch.stack(ps)
        module.load_state_dict(loaded, strict=False, assign=True)

    def _free_module(self, module):
        module.to_empty(device="meta")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- capture -----------------------------------------------------------------

    @torch.inference_mode()
    def prompt_end(self, prompts, thinking=False, system=None, gpu_batch=16, progress=None):
        """[len(prompts), n_layers + 1, hidden] in FP32 on CPU, like capture.prompt_end.

        Index 0 is the embedding output, index i+1 is layer i's output, each read
        at the prompt's true last token.
        """
        kw = {} if system is None else {"system": system}
        enc = self.tokenize([self.render(p, thinking=thinking, **kw) for p in prompts])
        input_ids = enc["input_ids"]
        attn2d = enc["attention_mask"]                      # [N, S], 1 = real token
        n, seq = input_ids.shape
        last = attn2d.sum(1) - 1                            # true last index per row
        pos = torch.arange(seq, device=self.device).unsqueeze(0)

        # Additive causal + padding mask for eager attention: [N, 1, S, S].
        neg = torch.finfo(torch.float32).min
        causal = torch.triu(torch.full((seq, seq), neg), diagonal=1)
        pad = torch.where(attn2d.bool(), 0.0, neg)[:, None, None, :]   # mask padded keys
        amask = (causal[None, None] + pad).to(self.device)

        # Embeddings.
        self._load_module(self.backbone.embeddings, "embeddings.")
        hidden = self.backbone.embeddings(input_ids.to(self.device)).to("cpu", self.dtype)
        self._free_module(self.backbone.embeddings)

        out = torch.empty(n, self.n_layers + 1, self.hidden_size, dtype=torch.float32)
        out[:, 0] = hidden[torch.arange(n), last.cpu()].float()

        # What each block's forward actually accepts (arg names vary by version).
        block0 = self.backbone.layers[0]
        accepts = set(inspect.signature(block0.forward).parameters)

        import time
        t0 = time.time()
        for i, block in enumerate(self.backbone.layers):
            tl = time.time()
            self._load_module(block, f"layers.{i}.")
            pieces = []
            for a in range(0, n, gpu_batch):
                b = slice(a, min(a + gpu_batch, n))
                kwargs = {}
                if "attention_mask" in accepts:
                    # Mamba uses the 2D mask (zeroes padded states); attention uses
                    # the 4D additive one. Pass whichever the layer reads.
                    kwargs["attention_mask"] = (amask[b] if _is_attention(block) else attn2d[b].to(self.device))
                if "position_ids" in accepts:
                    kwargs["position_ids"] = pos
                res = block(hidden[b].to(self.device), **kwargs)
                h = res[0] if isinstance(res, tuple) else res
                pieces.append(h.to("cpu", self.dtype))
            hidden = torch.cat(pieces)
            out[:, i + 1] = hidden[torch.arange(n), last.cpu()].float()
            kind = "attn" if _is_attention(block) else ("moe" if hasattr(block.mixer, "fc2_latent_proj") else "mamba")
            self._free_module(block)
            avail = guard(what=f"layer {i}")
            el = time.time() - t0
            eta = el / (i + 1) * (self.n_layers - i - 1)
            print(f"  layer {i:>2}/{self.n_layers} {kind:<5} {time.time()-tl:5.1f}s  "
                  f"(elapsed {el/60:.1f}m, eta {eta/60:.1f}m, avail {avail:.0f}GB)", flush=True)
            if progress:
                progress(i + 1, self.n_layers)
        return out


def _is_attention(block):
    """A NemotronH block whose mixer is self-attention (has q_proj)."""
    mixer = getattr(block, "mixer", None)
    return mixer is not None and hasattr(mixer, "q_proj")
