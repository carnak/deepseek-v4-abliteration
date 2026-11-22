"""The lab: the model in BF16 on one box, with its activations readable and its edits differentiable.

Replaces Heretic's role (measuring directions) in this project.

    model.py     load the model; render prompts exactly as the server does
    capture.py   read residual-stream activations, at the prompt's end or
                 across the tokens generated while reasoning
    edit.py      apply a candidate ablation as forward hooks
    directions.py  turn activations into directions, in the file format
                 tools/ablate.py reads
    cli.py       command line

Nothing here imports heretic.
"""
