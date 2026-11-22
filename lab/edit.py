"""A candidate ablation, applied as forward hooks.

A layer writes to the residual stream through three things: its attention
module, its routed experts and its shared expert. Removing a direction v from
every matrix one of them writes through, W -> W - w * v (v^T W), changes that
module's output y to y - w * v (v . y). So projecting the outputs is the same
function as editing the weights -- the equivalence heretic's fused-expert and
pre-ablation patches rely on, and that tools/ablate.py realises in the weights.

That makes an edit cheap to try (nothing is written or restored) and, because
the weights w are ordinary tensors, differentiable.

An Edit mirrors what tools/ablate.py can build, so anything found here can be
handed to it:

    basis[layer]    [k, hidden], orthonormal: the directions for that layer
    weight[layer]   {"attn": w, "shared": w, "experts": w}, each a float or a
                    tensor of shape [k] (one weight per direction)
"""
import torch

COMPONENTS = ("attn", "shared", "experts")


def orthonormal(vectors):
    """[k, hidden] -> orthonormal rows, in order (the first is the first, normalized)."""
    q, r = torch.linalg.qr(vectors.to(torch.float64).T)
    keep = r.diagonal().abs() > 1e-6
    return q.T[keep].to(torch.float32)


def component_modules(layer):
    """{component: module} for one decoder layer of a Qwen3.5/3.6 MoE model."""
    out = {}
    for name in ("self_attn", "linear_attn"):
        module = getattr(layer, name, None)
        if isinstance(module, torch.nn.Module):
            out["attn"] = module
    mlp = layer.mlp
    out["experts"] = mlp.experts
    # The shared expert's output is scaled by a per-token gate before it is
    # added. A scalar commutes with the projection, so hooking here is exact.
    out["shared"] = mlp.shared_expert
    return out


class Edit:
    def __init__(self):
        self.basis = {}   # layer -> [k, hidden]
        self.weight = {}  # layer -> {component: float | Tensor[k]}

    @classmethod
    def uniform(cls, direction_sets, layers, attn=1.0, shared=1.0, experts=1.0, scales=None):
        """What `ablate.py --layers a-b --attn .. --shared .. --experts .. --scales ..` builds.

        direction_sets: list of [layers + 1, hidden] tensors (index 0 is the
        embedding output, as in a residuals file).
        """
        edit = cls()
        for layer in layers:
            basis = orthonormal(torch.stack([d[layer + 1] for d in direction_sets]))
            s = torch.tensor(scales if scales else [1.0] * len(direction_sets), dtype=torch.float32)[: basis.shape[0]]
            edit.basis[layer] = basis
            edit.weight[layer] = {"attn": attn * s, "shared": shared * s, "experts": experts * s}
        return edit

    def install(self, lab):
        """Register the hooks. Returns a handle list; call remove() on each to undo."""
        handles = []
        for layer_index, basis in self.basis.items():
            modules = component_modules(lab.layers[layer_index])
            for component, module in modules.items():
                w = self.weight[layer_index].get(component, 0.0)
                handles.append(module.register_forward_hook(_projector(basis, w)))
        return handles


def _projector(basis, weight):
    def project(y):
        b = basis.to(y.device)
        w = weight.to(y.device) if torch.is_tensor(weight) else torch.full((b.shape[0],), float(weight), device=y.device)
        y32 = y.to(torch.float32)
        # (y . b_k) for each k, scaled, back along b_k.
        y32 = y32 - ((y32 @ b.T) * w) @ b
        return y32.to(y.dtype)

    def hook(module, args, output):
        # Attention modules may return (hidden_states, ...).
        if isinstance(output, tuple):
            return (project(output[0]), *output[1:])
        return project(output)

    return hook


class applied:
    """with applied(lab, edit): ...   -- the edit is in force inside the block only."""

    def __init__(self, lab, edit):
        self.lab, self.edit, self.handles = lab, edit, []

    def __enter__(self):
        if self.edit is not None:
            self.handles = self.edit.install(self.lab)
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        return False
