import os
import builtins


def print(*args, **kwargs):
    """Respect PRINT_ENABLED while mirroring the built-in print signature."""
    if os.getenv("PRINT_ENABLED", "1") == "1":
        builtins.print(*args, **kwargs)
