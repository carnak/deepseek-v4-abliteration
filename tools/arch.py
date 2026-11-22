"""Where a model architecture keeps the matrices that write to the residual stream.

Directional ablation edits exactly those matrices. The maths is the same for
every model; what differs is what the tensors are called, and whether a
quantized checkpoint stores MoE experts fused or one per expert. That naming is
all this file holds, so supporting another architecture means adding a class
here and nothing else.

A target is (component, kind, bf16 key, quantized name):
  kind "linear"  one matrix [out, in]; quantized under the same name.
  kind "fused"   BF16 holds [experts, out, in] in one tensor; the quantized
                 checkpoint holds one matrix per expert, named by expert_name().

COMPONENTS lists an arch's residual-writer kinds in a fixed order. ablate.py
builds one --<component> weight flag per entry and reads --place values in this
order, so the two archs below take different flags without ablate.py knowing
their names.
"""
import json
import os


class Qwen35Moe:
    """Qwen3.5 / Qwen3.6 MoE (transformers model_type qwen3_5_moe).

    A hybrid: some layers use standard self-attention, others Gated DeltaNet
    linear attention, never both. Every layer has a shared expert (an ordinary
    MLP) plus routed experts fused into a single parameter.
    """

    name = "qwen3_5_moe"
    COMPONENTS = ("attn", "shared", "experts")

    @staticmethod
    def num_layers(model_dir):
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        return cfg.get("text_config", cfg)["num_hidden_layers"]

    @staticmethod
    def prefix(layer):
        return f"model.language_model.layers.{layer}."

    @classmethod
    def targets(cls, layer, bf16):
        """Yield (component, kind, bf16_key, quantized_name) for one layer."""
        p = cls.prefix(layer)
        for attn in ("self_attn.o_proj", "linear_attn.out_proj"):
            if p + attn + ".weight" in bf16:
                yield "attn", "linear", p + attn + ".weight", p + attn
        yield "shared", "linear", p + "mlp.shared_expert.down_proj.weight", p + "mlp.shared_expert.down_proj"
        yield "experts", "fused", p + "mlp.experts.down_proj", p + "mlp.experts"

    @staticmethod
    def expert_name(quantized_name, expert):
        return f"{quantized_name}.{expert}.down_proj"


class NemotronH:
    """NVIDIA Nemotron-H (transformers model_type nemotron_h, inside nemotron_h_omni).

    88 layers, each exactly one of three block types, named in
    llm_config.layers_block_type:

      mamba (40)      residual writer mixer.out_proj           FP8
      attention (8)   residual writer mixer.o_proj             FP8
      moe (40)        mixer.fc2_latent_proj  (routed experts)  BF16, unquantized
                      mixer.shared_experts.down_proj (shared)  NVFP4

    Routed experts run in a 1024-wide latent space; fc2_latent_proj projects the
    combined expert output back to hidden, so editing that one BF16 matrix
    abliterates all 512 routed experts of a layer at once -- the per-expert
    tensors are never touched. The shared expert bypasses the latent space
    (hidden->hidden) and so is a separate writer. Every writer is [out=4096, in],
    so an output-side projection against a hidden-space direction applies cleanly.

    (Verified against the checkpoint when that model was abliterated.)
    """

    name = "nemotron_h"
    COMPONENTS = ("mamba", "attn", "moe_latent", "shared")

    @staticmethod
    def _llm_config(model_dir):
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        return cfg.get("llm_config", cfg)

    @classmethod
    def num_layers(cls, model_dir):
        # There is no num_hidden_layers; the layer count is the length of the
        # block-type list.
        return len(cls._llm_config(model_dir)["layers_block_type"])

    @staticmethod
    def prefix(layer):
        return f"language_model.backbone.layers.{layer}.mixer."

    @classmethod
    def targets(cls, layer, bf16):
        """Yield (component, kind, bf16_key, quantized_name) for one layer.

        The block type is read from tensor presence rather than config, so a
        layer yields only the writer(s) it actually has.
        """
        p = cls.prefix(layer)
        if p + "o_proj.weight" in bf16:                 # attention layer
            yield "attn", "linear", p + "o_proj.weight", p + "o_proj"
        elif p + "out_proj.weight" in bf16:             # mamba layer
            yield "mamba", "linear", p + "out_proj.weight", p + "out_proj"
        if p + "fc2_latent_proj.weight" in bf16:        # moe layer
            yield "moe_latent", "linear", p + "fc2_latent_proj.weight", p + "fc2_latent_proj"
            yield "shared", "linear", p + "shared_experts.down_proj.weight", p + "shared_experts.down_proj"


class DeepseekV4:
    """DeepSeek-V4 (model_type deepseek_v4), as DeepSeek ships it: native FP8/FP4.

    43 layers, hidden 4096, every layer attention + MoE (256 routed experts, top-6,
    plus one shared expert). Keys are DeepSeek's own names (layers.<i>.attn.wo_b),
    not transformers'.

      attention   attn.wo_b                     [4096, 8192]  FP8, 128x128 blocks
      shared      ffn.shared_experts.w2         [4096, 2048]  FP8, 128x128 blocks
      experts     ffn.experts.<e>.w2  (x256)    [4096, 2048]  FP4, 32-wide groups

    attn.wo_b is the second half of the grouped low-rank output projection
    (wo_a maps each head group to 1024, wo_b mixes the 8 groups to hidden), so it
    is the attention's only writer. Each expert is stored on its own, so a layer's
    routed experts are 256 targets of kind "linear".

    Hyper-connections: the residual is 4 copies of the hidden state. A sublayer
    reads a learned mix of them, and its one 4096-wide output is added to all 4,
    each scaled by its own weight. Removing v from what the writers output
    therefore removes it from all 4 copies at once -- the edit is the same as for
    a plain residual. Which state the direction is measured on is the lab's
    question (lab/stream_dsv4.py), not this file's.

    NATIVE_QUANT: there is no BF16 original. ablate.py decodes the edited matrix
    from the checkpoint itself, edits it, and re-encodes it in the same format.
    The DSpark draft head (mtp.*) is not edited: it reads the main model's hidden
    states and is not a writer of them.
    """

    name = "deepseek_v4"
    COMPONENTS = ("attn", "shared", "experts")
    NATIVE_QUANT = True

    @staticmethod
    def num_layers(model_dir):
        return json.load(open(os.path.join(model_dir, "config.json")))["num_hidden_layers"]

    @staticmethod
    def n_experts(model_dir):
        return json.load(open(os.path.join(model_dir, "config.json")))["n_routed_experts"]

    @staticmethod
    def prefix(layer):
        return f"layers.{layer}."

    @classmethod
    def targets(cls, layer, ckpt):
        """Yield (component, kind, source_key, quantized_name) for one layer."""
        p = cls.prefix(layer)
        yield "attn", "linear", p + "attn.wo_b.weight", p + "attn.wo_b"
        yield "shared", "linear", p + "ffn.shared_experts.w2.weight", p + "ffn.shared_experts.w2"
        e = 0
        while f"{p}ffn.experts.{e}.w2.weight" in ckpt:
            yield "experts", "linear", f"{p}ffn.experts.{e}.w2.weight", f"{p}ffn.experts.{e}.w2"
            e += 1


ARCHS = {a.name: a for a in (Qwen35Moe, NemotronH, DeepseekV4)}


def get(name):
    if name not in ARCHS:
        raise SystemExit(f"unknown --arch {name!r}; known: {', '.join(ARCHS)}. Add one in tools/arch.py.")
    return ARCHS[name]
