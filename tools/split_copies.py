#!/usr/bin/env python3
"""Write each hyper-connection copy's directions as a directions file of its own.

DeepSeek-V4's residual is 4 copies of the hidden state, and lab/stream_dsv4.py
measures a refusal direction on each copy as well as on their mean
(copy_directions [layers + 1, 4, hidden]). Later layers read a learned mix of
the copies, so the refusal they read lies in the span of the copies' directions.
Where the copies disagree (tools/geometry.py), removing that span, rather than
the mean's one direction, is the edit that removes refusal as read through any
mix. ablate.py takes several --directions and orthonormalizes them per layer,
so the span is four files:

    python3 split_copies.py directions/<name>.safetensors
    # -> directions/<name>-copy0.safetensors ... -copy3.safetensors
    ablate.py ... --directions <name>-copy0 ... --directions <name>-copy3
"""
import os
import sys

from safetensors.torch import load_file, save_file


def main():
    src = sys.argv[1]
    t = load_file(src)
    copies = t["copy_directions"]                        # [L+1, 4, H]
    stem = src[: -len(".safetensors")]
    for c in range(copies.shape[1]):
        out = f"{stem}-copy{c}.safetensors"
        save_file({"directions": copies[:, c].contiguous()}, out,
                  metadata={"source": os.path.basename(src), "copy": str(c)})
        print(out)


if __name__ == "__main__":
    main()
