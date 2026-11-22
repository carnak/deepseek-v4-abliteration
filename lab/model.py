"""Loading the model and rendering prompts.

Two things here exist because of specific failures:

- load() goes through the CPU. On unified memory the device-mapped loader keeps
  a second copy of every tensor it has moved, and a 65 GB model does not fit in
  121 GB that way.
- render() is the model's own chat template with its own thinking switch, and
  nothing appended. Heretic appends a "response prefix", and for a template that
  already opens a reasoning block that produced a prompt no server ever sends;
  a whole search was measured in that state.
"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

DEFAULT_SYSTEM = "You are a helpful assistant."


class Lab:
    def __init__(self, path, device="cuda"):
        self.path = path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding: the position that matters is the last one, and it must
        # be the same index for every prompt in a batch.
        self.tokenizer.padding_side = "left"

        config = AutoConfig.from_pretrained(path)
        cls = AutoModelForImageTextToText if hasattr(config, "vision_config") else AutoModelForCausalLM
        # Memory-mapped on the CPU side, then moved one parameter at a time.
        self.model = cls.from_pretrained(path, dtype="auto", device_map="cpu").to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    # --- structure ---------------------------------------------------------------

    @property
    def layers(self):
        inner = self.model.model
        return inner.language_model.layers if hasattr(inner, "language_model") else inner.layers

    @property
    def hidden_size(self):
        cfg = self.model.config
        return getattr(cfg, "text_config", cfg).hidden_size

    # --- prompts -----------------------------------------------------------------

    def render(self, prompt, thinking=False, system=DEFAULT_SYSTEM, messages=None):
        """The exact string the server builds for this request.

        With thinking off the Qwen3.6 template ends "<think>\\n\\n</think>\\n\\n";
        with it on, "<think>\\n". Nothing is added after it.
        """
        if messages is None:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=thinking
        )

    def tokenize(self, texts):
        # The template has already placed every special token.
        return self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


class OutOfHeadroom(RuntimeError):
    pass


def guard(min_available_gb=14.0, what="this step"):
    """Stop before the box does. Call inside any loop that allocates.

    On unified memory, running out is not a CUDA error that can be caught: the
    GPU and the host draw on one pool, and exhausting it starves the machine.
    PyTorch's allocator also keeps what it has freed, so a few long sequences
    ratchet usage up and it never comes back down by itself. So: hand cached
    blocks back, then check what the HOST has left, and raise while there is
    still room to raise in.

    Written after a training run on long traces took a box to 117 GB of 121
    without any error (one allocator warning, 80 steps earlier).
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) / 1024**2
                break
    if available < min_available_gb:
        raise OutOfHeadroom(
            f"{available:.1f} GB available after {what}, under the {min_available_gb:.0f} GB floor. "
            "Stopping rather than starving the box: use shorter sequences or a smaller batch."
        )
    return available


def memory_report(tag=""):
    """One line on where the box's memory is. AnonPages is the part that cannot be reclaimed."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            info[k] = int(v.split()[0]) / 1024**2
    used = info["MemTotal"] - info["MemAvailable"]
    cuda = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    print(f"[mem] {tag:<22} host used {used:6.1f} GB  available {info['MemAvailable']:6.1f}  "
          f"anon {info['AnonPages']:5.1f}  cuda {cuda:5.1f} (peak {peak:5.1f})", flush=True)
    return info["MemAvailable"]
