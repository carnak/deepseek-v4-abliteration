"""Reading the residual stream.

Two places to read it:

  prompt end   the last position of the rendered prompt, at every layer. This
               is what heretic measures, and what every direction used in the
               first-generation build came from.

  reasoning    the positions of the first N tokens the model generates with
               thinking on, averaged. The first build's remaining refusals are
               reached during reasoning, after the only position that had ever
               been looked at.

Both return a tensor [prompts, layers + 1, hidden] in FP32 on the CPU, where
index 0 is the embedding output and index i + 1 is layer i's output -- the
indexing heretic's residuals files use.
"""
import torch

from .model import batches, guard


@torch.inference_mode()
def prompt_end(lab, prompts, thinking=False, batch_size=32, system=None, progress=None):
    kw = {} if system is None else {"system": system}
    out = []
    for chunk in batches(prompts, batch_size):
        enc = lab.tokenize([lab.render(p, thinking=thinking, **kw) for p in chunk])
        hidden = lab.model(**enc, output_hidden_states=True).hidden_states
        # Left-padded, so the last position is the prompt's last token for every row.
        out.append(torch.stack([h[:, -1, :] for h in hidden], dim=1).to("cpu", torch.float32))
        if progress:
            progress(sum(o.shape[0] for o in out), len(prompts))
    return torch.cat(out)


END_OF_REASONING = "</think>"


@torch.inference_mode()
def trace_end(lab, pairs, batch_size=4, progress=None):
    """Residuals at the point where reasoning ends and the answer begins.

    pairs: [(prompt, trace)] where trace is what the model generated with
    thinking on. Each is replayed (teacher-forced) up to and including the end
    of its reasoning block and the whitespace after it, and the residual is read
    at that last position: the state from which the first word of the answer is
    produced. With thinking on, this is where a refusal has been decided -- the
    counterpart of the prompt's last token when there is no reasoning.

    Traces whose reasoning never finished are skipped. Returns
    ([n, layers + 1, hidden], indices of the pairs used).
    """
    texts, used = [], []
    for i, (prompt, trace) in enumerate(pairs):
        cut = trace.find(END_OF_REASONING)
        if cut < 0:
            continue
        # The template's own no-thinking form ends with the closing tag and a
        # blank line; match it.
        texts.append(lab.render(prompt, thinking=True) + trace[:cut] + END_OF_REASONING + "\n\n")
        used.append(i)
    out = []
    for chunk in batches(texts, batch_size):
        enc = lab.tokenize(chunk)
        hidden = lab.model(**enc, output_hidden_states=True).hidden_states
        out.append(torch.stack([h[:, -1, :] for h in hidden], dim=1).to("cpu", torch.float32))
        del hidden
        guard(what="replaying a trace")
        if progress:
            progress(sum(o.shape[0] for o in out), len(texts))
    if not out:
        raise SystemExit("no trace in this file finished its reasoning; generate longer traces")
    return torch.cat(out), used


@torch.inference_mode()
def reasoning(lab, prompts, tokens=32, skip=0, batch_size=16, progress=None):
    """Mean residual over generated reasoning tokens [skip, tokens), thinking on.

    Greedy generation, so the same prompt gives the same reasoning. The average
    is over positions the model produced itself; the prompt is not included.
    """
    out = []
    for chunk in batches(prompts, batch_size):
        enc = lab.tokenize([lab.render(p, thinking=True) for p in chunk])
        gen = lab.model.generate(
            **enc,
            max_new_tokens=tokens,
            min_new_tokens=tokens,   # do not let a row finish early and pad the average
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=lab.tokenizer.pad_token_id,
        )
        # hidden_states[step][layer] is [batch, positions fed at that step, hidden];
        # the last position of each step is the token just generated.
        steps = gen.hidden_states[skip:]
        per_step = [torch.stack([h[:, -1, :] for h in step], dim=1) for step in steps]
        out.append(torch.stack(per_step).mean(0).to("cpu", torch.float32))
        if progress:
            progress(sum(o.shape[0] for o in out), len(prompts))
    return torch.cat(out)
