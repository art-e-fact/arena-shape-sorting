# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Checks for ClipRecorder's checkpoint / cut semantics. Run: python test_demo_clips.py"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shape_sorting.demo_clips import ClipRecorder


class FakeRecorder:
    """Stand-in for SO101LeRobotRecorder: frames land in an episode only on save."""

    def __init__(self):
        self.pending: list = []
        self.episodes: list[list] = []

    def add_transition(self, observation, processed_action, *, task: str) -> None:
        assert task, "frames need a task string"
        self.pending.append(observation)

    def save_episode(self) -> None:
        assert self.pending, "refused to save an empty episode"
        self.episodes.append(self.pending)
        self.pending = []


def _add(clips: ClipRecorder, frames) -> None:
    for i in frames:
        clips.add(i, f"action{i}", "sort the shapes")


def test_success_saves_every_frame_in_order():
    rec = FakeRecorder()
    clips = ClipRecorder(rec)
    _add(clips, range(10))
    assert clips.save() == 10
    assert rec.episodes == [list(range(10))]
    assert rec.pending == []


def test_cut_drops_frames_after_the_checkpoint():
    rec = FakeRecorder()
    clips = ClipRecorder(rec)
    _add(clips, range(5))
    clips.checkpoint()
    _add(clips, range(5, 8))  # the mistake
    assert clips.cut() == 5
    assert rec.episodes == [[0, 1, 2, 3, 4]]

    _add(clips, range(8, 12))  # the recovery starts a new clip
    assert clips.save() == 4
    assert rec.episodes[1] == [8, 9, 10, 11]


def test_cut_without_a_checkpoint_saves_nothing():
    rec = FakeRecorder()
    clips = ClipRecorder(rec)
    _add(clips, range(6))
    assert clips.cut() == 0
    assert rec.episodes == []
    assert rec.pending == []


def test_confirmed_frames_are_dripped_not_burst():
    rec = FakeRecorder()
    clips = ClipRecorder(rec)
    _add(clips, range(40))
    clips.checkpoint()
    assert len(rec.pending) == 0, "nothing is fed before a checkpoint"
    _add(clips, range(40, 45))
    assert len(rec.pending) == 10, "5 adds x 2 confirmed frames each"
    assert clips.cut() == 40
    assert rec.episodes == [list(range(40))]


def test_cut_and_save_are_noops_without_frames():
    rec = FakeRecorder()
    clips = ClipRecorder(rec)
    clips.checkpoint()
    assert clips.cut() == 0
    assert clips.save() == 0
    assert rec.episodes == []


if __name__ == "__main__":
    for name, case in sorted(globals().items()):
        if name.startswith("test_"):
            case()
            print(f"ok {name}")
    print("all clip recorder checks passed")
