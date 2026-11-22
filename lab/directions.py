"""From activations to directions, in the file format tools/ablate.py reads.

A residuals file holds three tensors of shape [layers + 1, hidden]:
    good_means   mean residual over the harmless prompts
    bad_means    mean residual over the harmful prompts
    directions   unit refusal direction per position
Index 0 is the embedding output; layer i is at i + 1.
"""
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


def mean_difference(good, bad, orthogonalize=True):
    """good, bad: [prompts, layers + 1, hidden]. The direction heretic computes.

    orthogonalize=True is "projected abliteration": keep only the part of the
    difference that is orthogonal to the harmless mean, so that removing it does
    not also shrink what harmless prompts have in common.
    """
    good_means, bad_means = good.mean(0), bad.mean(0)
    directions = F.normalize(bad_means - good_means, p=2, dim=1)
    if orthogonalize:
        g = F.normalize(good_means, p=2, dim=1)
        directions = F.normalize(directions - (directions * g).sum(1, keepdim=True) * g, p=2, dim=1)
    return {"good_means": good_means, "bad_means": bad_means, "directions": directions}


def save(path, tensors, **metadata):
    save_file({k: v.to(torch.float32).contiguous() for k, v in tensors.items()}, path,
              metadata={k: str(v) for k, v in metadata.items()})


def load(path):
    return load_file(path)


def cosine_by_layer(a, b):
    """Per-position cosine between two [layers + 1, hidden] direction tensors."""
    return (F.normalize(a, dim=1) * F.normalize(b, dim=1)).sum(1)
