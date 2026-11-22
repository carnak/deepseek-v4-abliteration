"""Capture the residual stream of DeepSeek-V4 one layer at a time.

The checkpoint is 155 GiB of native FP8/FP4 (there is no BF16 original), which
does not fit in one 121 GB box. As lab/stream.py does for Nemotron, this builds
transformers' DeepseekV4Model on the meta device and runs the forward
"layer-outer": all prompts' activations are held at once, and for each of the 43
layers that layer's weights are read from the mmapped shards, decoded to BF16 on
the GPU, every prompt is pushed through, and the layer is freed. One layer in
BF16 is ~14 GB (256 experts dominate).

What is different from Nemotron:

  Names. The checkpoint uses DeepSeek's own names (layers.3.attn.wo_b);
  transformers' module uses its own (layers.3.self_attn.o_b_proj). hf_name()
  applies transformers' renaming for this model (conversion_mapping.py,
  "deepseek_v4"), and routed experts are stacked as transformers does: w1 and
  w3 concatenated into gate_up_proj, w2 into down_proj.

  Formats. A tensor with a sibling <name>.scale is decoded with
  tools/quant_formats.py's DeepSeek decoders (the same ones the build uses).
  Modules transformers keeps in FP32 (hyper-connection mixers, norms, sinks,
  compressor projections) are loaded in FP32; everything else BF16.

  Hyper-connections. The residual is 4 copies of the hidden state,
  [batch, seq, 4, hidden]. Captured at every layer boundary are all 4 copies;
  the direction is taken on their mean, which is what DeepSeek itself reads
  off the main model for its draft head (model.py: h.mean(dim=2)). The
  per-copy capture is kept so tools/geometry.py can check the copies agree.

  Rotary embeddings hold non-persistent buffers computed at construction, which
  a meta-device model never has; each layer's rotary modules are rebuilt on the
  GPU when the layer is loaded.

  Prompts are rendered by the checkpoint's encoding/encoding_dsv4.py, the
  model's own reference encoder, in "chat" mode (thinking off: the prompt ends
  "<｜Assistant｜></think>") or "thinking" mode. Verify against the server with
  `python -m lab.cli dsv4-check`, which compares token ids with vLLM's /tokenize
  and first-token distributions with the served model.

Batches are right-padded: a query only ever sees earlier tokens and compressed
windows that lie wholly before it, so padding after the last real token changes
nothing it reads.
"""
import json
import os
import re
import sys
import time
from collections import defaultdict

import torch
from safetensors import safe_open

from .model import DEFAULT_SYSTEM, guard

_TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
from quant_formats import dsv4_decode  # noqa: E402

# transformers' renaming for deepseek_v4 (conversion_mapping.py, 5.19), applied in order.
_RENAMES = [
    (r"\.attn\.", ".self_attn."),
    (r"\.ffn\.", ".mlp."),
    (r"\.indexer\.compressor\.", ".compressor.indexer."),
    (r"^embed\.weight$", "embed_tokens.weight"),
    (r"^head\.weight$", "lm_head.weight"),
    (r"^hc_head_fn$", "hc_head.hc_fn"),
    (r"^hc_head_base$", "hc_head.hc_base"),
    (r"^hc_head_scale$", "hc_head.hc_scale"),
    (r"\.attn_norm\.", ".input_layernorm."),
    (r"\.ffn_norm\.", ".post_attention_layernorm."),
    (r"\.hc_attn_fn$", ".attn_hc.fn"),
    (r"\.hc_attn_base$", ".attn_hc.base"),
    (r"\.hc_attn_scale$", ".attn_hc.scale"),
    (r"\.hc_ffn_fn$", ".ffn_hc.fn"),
    (r"\.hc_ffn_base$", ".ffn_hc.base"),
    (r"\.hc_ffn_scale$", ".ffn_hc.scale"),
    (r"\.attn_sink$", ".sinks"),
    (r"\.indexer\.weights_proj\.", ".compressor.indexer.scorer.weights_proj."),
    (r"\.indexer\.wq_b\.", ".compressor.indexer.q_b_proj."),
    (r"\.norm\.", ".kv_norm."),
    (r"\.ape$", ".position_bias"),
    (r"\.wq_a\.", ".q_a_proj."),
    (r"\.self_attn\.wq_b\.", ".self_attn.q_b_proj."),
    (r"\.wkv\.", ".kv_proj."),
    (r"\.wgate\.", ".gate_proj."),
    (r"\.wo_a\.", ".o_a_proj."),
    (r"\.wo_b\.", ".o_b_proj."),
    (r"\.q_norm\.", ".q_a_norm."),
    (r"\.gate\.bias$", ".gate.e_score_correction_bias"),
    (r"\.shared_experts\.w1\.", ".shared_experts.gate_proj."),
    (r"\.shared_experts\.w2\.", ".shared_experts.down_proj."),
    (r"\.shared_experts\.w3\.", ".shared_experts.up_proj."),
]
_EXPERT = re.compile(r"^(layers\.\d+\.mlp\.experts)\.(\d+)\.(w[123])\.weight$")

# transformers' _keep_in_fp32_modules(_strict) for this model.
_FP32_TOKENS = {"attn_hc", "ffn_hc", "hc_head", "sinks", "position_bias", "e_score_correction_bias",
                "q_a_norm", "kv_norm", "input_layernorm", "post_attention_layernorm", "norm"}
_FP32_PATHS = ("self_attn.compressor.kv_proj", "self_attn.compressor.gate_proj",
               "self_attn.compressor.indexer.kv_proj", "self_attn.compressor.indexer.gate_proj",
               "self_attn.compressor.indexer.scorer.weights_proj")


def hf_name(native):
    """A checkpoint key in transformers' naming, or None for the draft head (mtp.*)."""
    if native.startswith("mtp."):
        return None
    name = native
    for pat, rep in _RENAMES:
        name = re.sub(pat, rep, name)
    return name


def _keeps_fp32(name):
    return bool(_FP32_TOKENS & set(name.split("."))) or any(p in name for p in _FP32_PATHS)


class StreamDSV4Lab:
    # FP32 throughout, not BF16. transformers keeps the norms in FP32 and its
    # RMSNorm returns weight * x, so a BF16 model with FP32 norms hands FP32 to
    # BF16 projections and fails; FP32 everywhere also makes this the most exact
    # reference to hold the served model against. One layer is ~26 GB in FP32.
    def __init__(self, model_dir, device="cuda", dtype=torch.float32, reasoning_effort=None):
        from transformers import AutoConfig, AutoTokenizer
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Model

        self.model_dir = model_dir
        self.device = device
        self.dtype = dtype
        self.reasoning_effort = reasoning_effort

        sys.path.insert(0, os.path.join(model_dir, "encoding"))
        import encoding_dsv4  # the checkpoint's own reference encoder
        self.encoding = encoding_dsv4
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        eos = self.tokenizer.convert_tokens_to_ids(encoding_dsv4.eos_token)
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos

        self.config = AutoConfig.from_pretrained(model_dir)
        if hasattr(self.config, "quantization_config"):
            del self.config.quantization_config   # weights are decoded here, not by transformers
        self.config._attn_implementation = "eager"
        with torch.device("meta"):
            self.model = DeepseekV4Model(self.config)
        self.model.eval()
        self.n_layers = self.config.num_hidden_layers
        self.hc = self.config.hc_mult

        idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json"), encoding="utf-8"))
        self.where = {}               # hf name -> (file, native key)
        self.experts = defaultdict(dict)   # "layers.3.mlp.experts" -> {(e, w): (file, native key)}
        self.native = {}              # native key -> file
        for key, shard in idx["weight_map"].items():
            f = os.path.join(model_dir, shard)
            self.native[key] = f
            name = hf_name(key)
            if name is None or key.endswith(".scale"):
                continue
            m = _EXPERT.match(name)
            if m:
                self.experts[m.group(1)][(int(m.group(2)), m.group(3))] = (f, key)
            else:
                self.where[name] = (f, key)

    @property
    def hidden_size(self):
        return self.config.hidden_size

    def __len__(self):
        return self.n_layers

    # --- prompts --------------------------------------------------------------------

    def render(self, prompt, thinking=False, system=DEFAULT_SYSTEM, messages=None):
        if messages is None:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        return self.encoding.encode_messages(
            messages, thinking_mode="thinking" if thinking else "chat",
            reasoning_effort=self.reasoning_effort)

    def tokenize_one(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    # --- weights --------------------------------------------------------------------

    def _read(self, f, key, handle):
        """One checkpoint tensor, decoded if it has a .scale sibling, on the GPU."""
        t = handle(f).get_tensor(key)
        skey = key[: -len(".weight")] + ".scale" if key.endswith(".weight") else None
        if skey and skey in self.native:
            s = handle(self.native[skey]).get_tensor(skey)
            t = dsv4_decode(t.to(self.device), s.to(self.device))
        return t.to(self.device)

    def _load_module(self, module, prefix):
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding

        opened = {}

        def handle(f):
            if f not in opened:
                opened[f] = safe_open(f, framework="pt", device="cpu")
            return opened[f]

        loaded = {}
        names = [n for n, _ in module.named_parameters()] + [n for n, _ in module.named_buffers()]
        for local in names:
            full = prefix + local
            if full.endswith("mlp.experts.gate_up_proj") or full.endswith("mlp.experts.down_proj"):
                # Filled in place: stacking a list would hold the layer twice.
                base = full.rsplit(".", 1)[0]
                table = self.experts[base]
                n, inter, hid = self.config.n_routed_experts, self.config.moe_intermediate_size, self.hidden_size
                if full.endswith("gate_up_proj"):
                    t = torch.empty((n, 2 * inter, hid), dtype=self.dtype, device=self.device)
                    for e in range(n):
                        t[e, :inter] = self._read(*table[(e, "w1")], handle)
                        t[e, inter:] = self._read(*table[(e, "w3")], handle)
                else:
                    t = torch.empty((n, hid, inter), dtype=self.dtype, device=self.device)
                    for e in range(n):
                        t[e] = self._read(*table[(e, "w2")], handle)
                loaded[local] = t
            elif full in self.where:
                t = self._read(*self.where[full], handle)
                if t.is_floating_point():
                    t = t.to(torch.float32 if _keeps_fp32(full) else self.dtype)
                loaded[local] = t
            elif "rotary_emb" in local or local.endswith("_inv_freq"):
                continue          # non-persistent; rebuilt below
            else:
                raise KeyError(f"no checkpoint tensor for {full}")
        module.load_state_dict(loaded, strict=False, assign=True)

        # Rotary embeddings: rebuild on the GPU (their buffers are computed, not stored).
        for parent in list(module.modules()):
            for child_name, child in list(parent.named_children()):
                if isinstance(child, DeepseekV4RotaryEmbedding):
                    setattr(parent, child_name, DeepseekV4RotaryEmbedding(self.config, device=self.device))
        left = [n for n, p in list(module.named_parameters()) + list(module.named_buffers()) if p.is_meta]
        if left:
            raise RuntimeError(f"{prefix}: still on meta after loading: {left[:5]}")

    def _free_module(self, module):
        module.to_empty(device="meta")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- forward --------------------------------------------------------------------

    def _mask(self, seq):
        """Causal sliding-window mask, additive [1, 1, S, S]: key k is visible to query q
        when q - window < k <= q (transformers' sliding_window_causal semantics)."""
        q = torch.arange(seq, device=self.device)[:, None]
        k = torch.arange(seq, device=self.device)[None, :]
        ok = (k <= q) & (k > q - self.config.sliding_window)
        neg = torch.finfo(self.dtype).min
        return torch.where(ok, 0.0, neg).to(self.dtype)[None, None]

    def _batches(self, texts, gpu_batch):
        """Length-sorted batches, each right-padded only to its own longest prompt.

        Returns [(row indices, input_ids [b, s], last [b])]."""
        ids = [self.tokenize_one(t) for t in texts]
        order = sorted(range(len(ids)), key=lambda r: len(ids[r]))
        out = []
        for a in range(0, len(order), gpu_batch):
            rows = order[a: a + gpu_batch]
            seq = max(len(ids[r]) for r in rows)
            x = torch.full((len(rows), seq), self.pad_id, dtype=torch.long)
            for k, r in enumerate(rows):
                x[k, : len(ids[r])] = torch.tensor(ids[r])
            out.append((torch.tensor(rows), x, torch.tensor([len(ids[r]) - 1 for r in rows])))
        return out

    @torch.inference_mode()
    def run(self, texts, gpu_batch=8, progress=None, logits_top=0):
        """Push texts through every layer. Returns (means, streams, top):
        means   [N, layers + 1, hidden] FP32, the mean of the 4 copies at each
                prompt's last token (index 0: the embedding, layer i at i + 1)
        streams [N, layers + 1, 4, hidden] FP32, the copies themselves
        top     if logits_top: per prompt {token id: logprob} for the top tokens
                of the next-token distribution at the last token, else None
        """
        batches = self._batches(texts, gpu_batch)
        n = len(texts)
        means = torch.empty(n, self.n_layers + 1, self.hidden_size, dtype=torch.float32)
        streams = torch.empty(n, self.n_layers + 1, self.hc, self.hidden_size, dtype=torch.float32)

        def record(index, hidden):
            for (rows, _, last), h in zip(batches, hidden):
                streams[rows, index] = h[torch.arange(len(rows)), last].float()
            means[:, index] = streams[:, index].mean(1)

        # hidden[k]: batch k's 4 copies, [b, s, 4, hidden], kept on the CPU between layers.
        self._load_module(self.model.embed_tokens, "embed_tokens.")
        hidden = []
        for _, x, _ in batches:
            e = self.model.embed_tokens(x.to(self.device)).to("cpu", self.dtype)
            hidden.append(e.unsqueeze(2).expand(-1, -1, self.hc, -1).contiguous())
        self._free_module(self.model.embed_tokens)
        record(0, hidden)

        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding
        rot = DeepseekV4RotaryEmbedding(self.config, device=self.device)

        t0 = time.time()
        for i, block in enumerate(self.model.layers):
            tl = time.time()
            self._load_module(block, f"layers.{i}.")
            for k, (_, x, _) in enumerate(batches):
                seq = x.shape[1]
                pos = torch.arange(seq, device=self.device).unsqueeze(0)
                h = hidden[k].to(self.device)
                pe = {"main": rot(h, position_ids=pos, layer_type="main"),
                      "compress": rot(h, position_ids=pos, layer_type="compress")}
                # Per row: the compressor concatenates a per-query bias of shape [b, 1, S, T].
                mask = self._mask(seq).expand(h.shape[0], -1, -1, -1)
                out = block(h, position_embeddings=pe, position_ids=pos, attention_mask=mask,
                            input_ids=x.to(self.device), past_key_values=None)
                hidden[k] = out.to("cpu", self.dtype)
            record(i + 1, hidden)
            self._free_module(block)
            avail = guard(what=f"layer {i}")
            el = time.time() - t0
            print(f"  layer {i:>2}/{self.n_layers} {self.config.layer_types[i][:7]:<7} "
                  f"{time.time() - tl:5.1f}s  (elapsed {el / 60:.1f}m, eta "
                  f"{el / (i + 1) * (self.n_layers - i - 1) / 60:.1f}m, avail {avail:.0f}GB)", flush=True)
            if progress:
                progress(i + 1, self.n_layers)

        top = None
        if logits_top:
            for mod, pre in ((self.model.hc_head, "hc_head."), (self.model.norm, "norm.")):
                self._load_module(mod, pre)
            f, key = self.where["lm_head.weight"] if "lm_head.weight" in self.where else (None, None)
            if f is None:
                raise KeyError("no lm_head.weight (head.weight) in the checkpoint")
            with safe_open(f, framework="pt", device="cpu") as h:
                head = h.get_tensor(key).to(self.device, self.dtype)
            top = [None] * n
            for (rows, _, last), hk in zip(batches, hidden):
                final = hk[torch.arange(len(rows)), last].to(self.device)      # [b, 4, hidden]
                x = self.model.norm(self.model.hc_head(final.unsqueeze(1))).squeeze(1)
                logp = torch.log_softmax((x.to(self.dtype) @ head.T).float(), dim=-1)
                vals, ids = logp.topk(logits_top, dim=-1)
                for k, r in enumerate(rows.tolist()):
                    top[r] = {int(t): float(v) for t, v in zip(ids[k].tolist(), vals[k].tolist())}
            del head
            self._free_module(self.model.hc_head)
            self._free_module(self.model.norm)
        return means, streams, top

    @torch.inference_mode()
    def screen(self, texts, configs, gpu_batch=32, logits_top=20):
        """Next-token distributions under several candidate edits, in one streaming pass.

        configs: [(name, {layer: (basis [k, hidden], weight)})]. Each edit is applied
        exactly, as a projection on what the attention and the MoE write (x -> x -
        w * (x B^T) B, B orthonormal) -- the same edit ablate.py makes in the weights,
        without the re-encoding. Every config keeps its own activations; each layer's
        weights are read once for all of them. Returns {name: [{token id: logprob}]}.
        """
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding

        batches = self._batches(texts, gpu_batch)
        current = {"edit": None}

        def project(out):
            e = current["edit"]
            if e is None:
                return out
            b, w = e
            return out - w * (out.to(b.dtype) @ b.T @ b).to(out.dtype)

        def attn_hook(_m, _i, out):
            return (project(out[0]),) + tuple(out[1:])

        def mlp_hook(_m, _i, out):
            return project(out)

        self._load_module(self.model.embed_tokens, "embed_tokens.")
        start = []
        for _, x, _ in batches:
            e = self.model.embed_tokens(x.to(self.device)).to("cpu", self.dtype)
            start.append(e.unsqueeze(2).expand(-1, -1, self.hc, -1).contiguous())
        self._free_module(self.model.embed_tokens)
        hidden = {name: [h.clone() for h in start] for name, _ in configs}
        rot = DeepseekV4RotaryEmbedding(self.config, device=self.device)

        t0 = time.time()
        for i, block in enumerate(self.model.layers):
            self._load_module(block, f"layers.{i}.")
            hooks = [block.self_attn.register_forward_hook(attn_hook), block.mlp.register_forward_hook(mlp_hook)]
            for name, edits in configs:
                e = edits.get(i)
                current["edit"] = None if e is None else (e[0].to(self.device, torch.float32), float(e[1]))
                for k, (_, x, _) in enumerate(batches):
                    seq = x.shape[1]
                    pos = torch.arange(seq, device=self.device).unsqueeze(0)
                    h = hidden[name][k].to(self.device)
                    pe = {"main": rot(h, position_ids=pos, layer_type="main"),
                          "compress": rot(h, position_ids=pos, layer_type="compress")}
                    mask = self._mask(seq).expand(h.shape[0], -1, -1, -1)
                    out = block(h, position_embeddings=pe, position_ids=pos, attention_mask=mask,
                                input_ids=x.to(self.device), past_key_values=None)
                    hidden[name][k] = out.to("cpu", self.dtype)
            current["edit"] = None
            for hk in hooks:
                hk.remove()
            self._free_module(block)
            guard(what=f"layer {i}")
            print(f"  screen layer {i:>2}/{self.n_layers}  ({(time.time() - t0) / 60:.1f}m)", flush=True)

        for mod, pre in ((self.model.hc_head, "hc_head."), (self.model.norm, "norm.")):
            self._load_module(mod, pre)
        f, key = self.where["lm_head.weight"]
        with safe_open(f, framework="pt", device="cpu") as hf:
            head = hf.get_tensor(key).to(self.device, self.dtype)
        result = {}
        for name, _ in configs:
            top = [None] * len(texts)
            for (rows, _, last), hk in zip(batches, hidden[name]):
                final = hk[torch.arange(len(rows)), last].to(self.device)
                xx = self.model.norm(self.model.hc_head(final.unsqueeze(1))).squeeze(1)
                logp = torch.log_softmax((xx.to(self.dtype) @ head.T).float(), dim=-1)
                vals, ids = logp.topk(logits_top, dim=-1)
                for k, r in enumerate(rows.tolist()):
                    top[r] = {int(t): float(v) for t, v in zip(ids[k].tolist(), vals[k].tolist())}
            result[name] = top
        del head
        self._free_module(self.model.hc_head)
        self._free_module(self.model.norm)
        return result

    def prompt_end(self, prompts, thinking=False, system=None, gpu_batch=8, progress=None):
        """[len(prompts), layers + 1, hidden], like StreamLab.prompt_end; the per-copy
        capture is left in self.last_streams."""
        kw = {} if system is None else {"system": system}
        means, streams, _ = self.run([self.render(p, thinking=thinking, **kw) for p in prompts],
                                     gpu_batch=gpu_batch, progress=progress)
        self.last_streams = streams
        return means
