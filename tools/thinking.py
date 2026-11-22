"""The chat_template_kwargs that switch a served model's reasoning on or off.

Templates disagree on the name: Qwen and Nemotron read `enable_thinking`,
DeepSeek-V4 reads `thinking`. A server ignores a key its template does not use,
so sending the wrong one is silent: against DeepSeek, {"enable_thinking": false}
still renders "<think>" and every "thinking off" measurement would really be
taken with thinking on (checked with vLLM's /tokenize, 2026-11-20).

The name comes from THINK_KWARG (models/<model>.env; box/run_queue.sh exports
it), default enable_thinking. Standard library only, like the tools using it.
"""
import os


def kwargs(on):
    return {os.environ.get("THINK_KWARG", "enable_thinking"): bool(on)}
