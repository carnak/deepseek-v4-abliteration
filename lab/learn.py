"""Learning the edit instead of sweeping it (an experiment on an earlier model; not used for DeepSeek-V4).

Generation 1 chose three numbers by hand (a weight for attention, for the MLP
matrices, and a scale for the second direction) and applied them to all 28
layers alike. Here every layer has its own weight per component and per
direction, and they are found by gradient descent.

What it descends on. With thinking on, the build's remaining refusals are
decided during the reasoning, and no single direction read there separates them
(queues 08 and 09). So the signal is not a direction but the behaviour itself,
teacher-forced:

    resisted   reasoning the model produced on its way to refusing a request
    complied   reasoning it produced on its way to answering one
    harmless   ordinary prompts

    loss =   mean log p(resisted trace)         make these less likely ...
           - keep * mean log p(complied trace)  ... keep these likely ...
           + drift * KL(original || edited)     ... and stay close on harmless prompts

The first term is capped: once a refusing trace costs `cap` nats per token it
stops counting, so the optimizer cannot buy the loss down by wrecking the model
on those prompts. The edit has a few hundred parameters, all scaling
projections along directions already known to be about refusal; it has no
capacity to do much else.

The loss is a stand-in. A model can be pushed off its refusing trace without
landing on a compliant one. Nothing learned here is believed until it has been
built into the weights, served and judged (tools/ablate.py --weights-file).
"""
import json
import time

import torch
import torch.nn.functional as F

from .edit import COMPONENTS, Edit, applied
from .model import batches, guard


def learnable(edit, device):
    """Turn an Edit's weights into parameters. Returns the list to optimize."""
    params = []
    for layer in edit.basis:
        k = edit.basis[layer].shape[0]
        for c in COMPONENTS:
            w = edit.weight[layer].get(c, 0.0)
            w = w.clone().float() if torch.is_tensor(w) else torch.full((k,), float(w))
            p = w.to(device).requires_grad_(True)
            edit.weight[layer][c] = p
            params.append(p)
    return params


def export(edit):
    """{layer: {component: [weight per direction]}}, for ablate.py --weights-file."""
    return {str(layer): {c: [round(float(x), 4) for x in edit.weight[layer][c].detach().cpu()] for c in COMPONENTS}
            for layer in sorted(edit.basis)}


END_OF_REASONING = "</think>"


def encode_traces(lab, pairs, max_tokens, window=None, after=64, max_context=900):
    """[(prompt, trace)] -> list of (input_ids, index of the first scored token), thinking on.

    window=None scores the first max_tokens tokens of each trace. That was the
    first attempt and it found nothing, because the model reasons for hundreds
    of tokens and decides late.

    With a window, only the decision is scored: the last `window` tokens ending
    `after` tokens into the answer. Everything before is context the model is
    given, not something it is graded on. Traces whose reasoning never finished
    are dropped, as are ones too long to replay within max_context tokens.
    """
    out = []
    for prompt, trace in pairs:
        head = lab.tokenizer(lab.render(prompt, thinking=True), add_special_tokens=False)["input_ids"]
        if window is None:
            tail = lab.tokenizer(trace, add_special_tokens=False)["input_ids"][:max_tokens]
            if len(tail) >= 8:
                out.append((head + tail, len(head)))
            continue
        cut = trace.find(END_OF_REASONING)
        if cut < 0:
            continue
        reasoning = lab.tokenizer(trace[: cut + len(END_OF_REASONING)], add_special_tokens=False)["input_ids"]
        answer = lab.tokenizer(trace[cut + len(END_OF_REASONING):], add_special_tokens=False)["input_ids"][:after]
        ids = head + reasoning + answer
        if len(ids) > max_context or len(answer) < 8:
            continue
        out.append((ids, max(len(head), len(ids) - window)))
    return out


def trace_nll(lab, items):
    """Mean negative log-likelihood per continuation token, one value per trace."""
    longest = max(len(ids) for ids, _ in items)
    pad = lab.tokenizer.pad_token_id
    input_ids = torch.full((len(items), longest), pad, dtype=torch.long)
    mask = torch.zeros((len(items), longest), dtype=torch.long)
    labels = torch.full((len(items), longest), -100, dtype=torch.long)
    for i, (ids, n_prompt) in enumerate(items):
        offset = longest - len(ids)  # left-padded, like everything else here
        input_ids[i, offset:] = torch.tensor(ids)
        mask[i, offset:] = 1
        labels[i, offset + n_prompt:] = torch.tensor(ids[n_prompt:])
    input_ids, mask, labels = (t.to(lab.device) for t in (input_ids, mask, labels))
    logits = lab.model(input_ids=input_ids, attention_mask=mask).logits[:, :-1].float()
    targets = labels[:, 1:]
    per_token = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                ignore_index=-100, reduction="none").view(targets.shape)
    counts = (targets != -100).sum(1).clamp(min=1)
    return per_token.sum(1) / counts


def last_token_logprobs(lab, prompts, thinking=False):
    enc = lab.tokenize([lab.render(p, thinking=thinking) for p in prompts])
    return F.log_softmax(lab.model(**enc).logits[:, -1].float(), dim=-1)


def train(lab, edit, resisted, complied, harmless, steps=200, lr=0.02, batch=4,
          keep=1.0, drift=1.0, cap=4.0, log_every=10, seed=0, log=print):
    """Optimize the edit's weights in place. Returns the per-step log."""
    gen = torch.Generator().manual_seed(seed)
    params = learnable(edit, lab.device)
    opt = torch.optim.Adam(params, lr=lr)

    # What "close to the original" means: the unedited model's next-token
    # distribution on harmless prompts, with thinking off and on.
    log(f"reference distributions for {len(harmless)} harmless prompts...")
    reference = {}
    with torch.inference_mode():
        for thinking in (False, True):
            reference[thinking] = torch.cat([last_token_logprobs(lab, chunk, thinking).cpu()
                                             for chunk in batches(harmless, 16)])

    def pick(pool, n):
        idx = torch.randperm(len(pool), generator=gen)[:n].tolist()
        return idx, [pool[i] for i in idx]

    history = []
    t0 = time.time()
    with applied(lab, edit):
        for step in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            with torch.enable_grad():
                _, r = pick(resisted, batch)
                nll_r = trace_nll(lab, r)
                # Negative NLL: descending this raises the cost of refusing
                # traces, up to the cap and no further.
                loss_r = -nll_r.clamp(max=cap).mean()
                loss_r.backward()

                _, c = pick(complied, batch)
                nll_c = trace_nll(lab, c)
                (keep * nll_c.mean()).backward()
                r_value, c_value = nll_r.mean().item(), nll_c.mean().item()

                idx, h = pick(harmless, batch)
                thinking = bool(step % 2)
                edited = last_token_logprobs(lab, h, thinking)
                ref = reference[thinking][idx].to(lab.device)
                kl = F.kl_div(edited, ref, log_target=True, reduction="batchmean")
                (drift * kl).backward()
            opt.step()
            # Long traces make the allocator hoard; this both returns the cache
            # and aborts the run if the host is short of memory.
            del nll_r, nll_c, edited, ref, loss_r
            available = guard(what=f"training step {step}")

            row = {"step": step, "nll_resisted": r_value, "nll_complied": c_value, "kl": kl.item(),
                   "available_gb": round(available, 1)}
            history.append(row)
            if step == 1 or step % log_every == 0:
                recent = history[-log_every:]
                avg = {k: sum(x[k] for x in recent) / len(recent) for k in ("nll_resisted", "nll_complied", "kl")}
                log(f"step {step:>4}  nll resisted {avg['nll_resisted']:.3f}  complied {avg['nll_complied']:.3f}  "
                    f"kl {avg['kl']:.4f}  ({time.time() - t0:.0f}s)")
    return history


@torch.inference_mode()
def generate_traces(lab, prompts, tokens=256, batch_size=16, progress=None):
    """What the model writes, thinking on, greedy: the raw text after the prompt."""
    out = []
    for chunk in batches(prompts, batch_size):
        enc = lab.tokenize([lab.render(p, thinking=True) for p in chunk])
        gen = lab.model.generate(**enc, max_new_tokens=tokens, do_sample=False,
                                 pad_token_id=lab.tokenizer.pad_token_id)
        texts = lab.tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=False)
        # Drop the padding a finished row was filled with, keep everything else
        # (including </think>, which is part of what is being scored).
        pad = lab.tokenizer.pad_token
        out.extend(t.replace(pad, "") if pad else t for t in texts)
        guard(what="generating traces")
        if progress:
            progress(len(out), len(prompts))
    return out


def save_traces(path, prompts, traces):
    with open(path, "w", encoding="utf-8") as f:
        for p, t in zip(prompts, traces):
            f.write(json.dumps({"prompt": p, "trace": t}, ensure_ascii=False) + "\n")


def load_traces(path, limit=None):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    return [(r["prompt"], r["trace"]) for r in rows][:limit]
