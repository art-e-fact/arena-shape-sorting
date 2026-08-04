# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Optional keyboard truncation for debugging episode resets."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

_RESET_REQUESTED = False
_LISTENER: _DebugKeyResetListener | None = None


class _DebugKeyResetListener:
    """Subscribe to a Kit keyboard key and set a one-shot episode-end flag."""

    def __init__(self, key_name: str = "R"):
        import carb
        import omni.appwindow

        self._key_name = key_name
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_event(event),
        )

    def _on_event(self, event) -> bool:
        import carb

        global _RESET_REQUESTED
        if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input.name == self._key_name:
            _RESET_REQUESTED = True
        return True

    def shutdown(self) -> None:
        if self._sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            self._sub = None


def _ensure_listener() -> None:
    global _LISTENER
    if _LISTENER is None:
        _LISTENER = _DebugKeyResetListener(key_name="R")
        print("[shape_sorting] debug_key_reset enabled — press R (Isaac window focused) to end the episode")


def debug_key_reset_termination(env) -> torch.Tensor:
    """Truncation term: True for all envs when R was pressed since the last check."""
    import torch

    global _RESET_REQUESTED
    _ensure_listener()
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if _RESET_REQUESTED:
        done[:] = True
        _RESET_REQUESTED = False
        print("[shape_sorting] R pressed — ending episode")
    return done
