import os
import builtins


def print(*args, **kwargs):
    """Skip printing when CUDA graph mode is enabled."""
    if os.getenv("CUDAGRAPH", "0") != "1":
        builtins.print(*args, **kwargs)
