"""SO-101 for Isaac Lab Arena.

Call :func:`register` after ``SimulationApp`` starts (imports Isaac Lab).
"""

from __future__ import annotations

_REGISTERED = False


def register() -> None:
    """Import modules so ``@register_asset`` / device / retargeter decorators run."""
    global _REGISTERED
    if _REGISTERED:
        return
    from arena_so101 import devices as _devices  # noqa: F401
    from arena_so101 import retargeters as _retargeters  # noqa: F401
    from arena_so101.embodiments import so101 as _so101  # noqa: F401

    _REGISTERED = True
