# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Split one scripted rollout into several LeRobot episodes ("clips").

A rollout is no longer one episode. The policy marks *checkpoints* after verified
sub-steps and asks for a *cut* when it detects its own mistake; :class:`ClipRecorder`
keeps frames in RAM until they are confirmed, so a cut can throw away the mistake
while keeping everything before it as an episode of its own.

Contract, per env step (frame ``t`` is the observation the policy acted on plus the
action that was applied). Events are drained *after* ``get_action`` and *before*
``add`` of that step:

``checkpoint()``
    Every buffered frame is safe to keep.
``cut()``
    Drop frames recorded since the last checkpoint; save the confirmed prefix as an
    episode if it is non-empty. Frame ``t`` becomes frame 0 of the next clip.
``save()``
    Task succeeded — confirm everything, then close the clip.

``["checkpoint", "cut"]`` in the same step is a split that drops nothing, which is what
a correction that must *keep* the bad approach (labelled with a retry) will need.

Why the RAM buffer: LeRobot's streaming encoder has no partial-truncation API — frames
fed to it are either a whole episode or discarded wholesale — and its per-camera queue
holds 30 frames, dropping anything beyond that with only a warning. So confirmed frames
are handed over a few per step (see ``_DRIP``) instead of in one burst.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Protocol


class _Recorder(Protocol):
    """The part of ``SO101LeRobotRecorder`` this wrapper uses."""

    def add_transition(self, observation: Any, processed_action: Any, *, task: str) -> None: ...

    def save_episode(self) -> None: ...


class ClipRecorder:
    """Buffer frames until confirmed, then commit them as one episode per clip."""

    _DRIP = 2
    """Confirmed frames handed to the recorder per :meth:`add`.

    Must be > 1, or a confirmed backlog never drains (one frame arrives per step).
    2 keeps the encoder queue ~15 steps from full while a fresh episode's encoder
    pays its codec-init cost on the first frame.
    """

    def __init__(self, recorder: _Recorder):
        self._recorder = recorder
        self._buffer: deque[tuple[Any, Any, str]] = deque()
        self._confirmed = 0  # leading buffered frames that are safe to keep
        self._fed = 0  # frames of the current clip already handed to the recorder

    def add(self, observation: Any, action: Any, task: str) -> None:
        """Buffer one frame, then drip confirmed frames into the recorder."""
        self._buffer.append((observation, action, task))
        for _ in range(min(self._DRIP, self._confirmed)):
            self._feed()

    def checkpoint(self) -> None:
        """Mark every buffered frame as safe to keep."""
        self._confirmed = len(self._buffer)

    def cut(self) -> int:
        """Drop unconfirmed frames; save the confirmed prefix. Returns frames saved."""
        while self._confirmed:
            self._feed()
        saved, self._fed = self._fed, 0
        self._buffer.clear()
        if saved:
            self._recorder.save_episode()
        return saved

    def save(self) -> int:
        """Confirm everything buffered and close the clip. Returns frames saved."""
        self.checkpoint()
        return self.cut()

    def _feed(self) -> None:
        observation, action, task = self._buffer.popleft()
        self._recorder.add_transition(observation, action, task=task)
        self._confirmed -= 1
        self._fed += 1
