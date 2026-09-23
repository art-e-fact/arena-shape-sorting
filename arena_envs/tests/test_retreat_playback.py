# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Checks that UP retraces a GO descent backwards. Run: python test_retreat_playback.py

Needs torch + arena_so101 (the Arena venv), but no Isaac Sim app.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shape_sorting.curobo_policy import CuroboPolicy, CuroboPolicyCfg

_DEVICE = torch.device("cpu")


class FakeMotion:
    """Only the two calls trajectory playback makes."""

    def notify_joint(self, q) -> None:
        pass

    def q_to_action(self, q, device, *, jaw: float) -> torch.Tensor:
        return torch.cat([q.to(device), torch.tensor([jaw], device=device)])


def _play(policy, forward: bool, jaw: float) -> list[float]:
    step = policy._playback_traj if forward else policy._playback_traj_reversed
    out = []
    while (action := step(_DEVICE, FakeMotion(), jaw=jaw)) is not None:
        assert abs(float(action[-1]) - jaw) < 1e-6
        out.append(float(action[0]))
    return out


def _policy_with_traj(waypoints: list[float]) -> CuroboPolicy:
    policy = CuroboPolicy(CuroboPolicyCfg())
    policy._traj = torch.tensor(waypoints, dtype=torch.float32).unsqueeze(1)
    policy._step_idx = 0
    return policy


def test_retreat_walks_the_descent_backwards():
    policy = _policy_with_traj([10.0, 11.0, 12.0, 13.0, 14.0])
    assert _play(policy, forward=True, jaw=-0.17) == [10.0, 11.0, 12.0, 13.0, 14.0]
    # Starts where the descent ended, so the arm never jumps on the first retreat step.
    assert _play(policy, forward=False, jaw=1.74) == [14.0, 13.0, 12.0, 11.0, 10.0]


def test_retreat_ends_back_at_the_hover_pose():
    policy = _policy_with_traj([10.0, 11.0, 12.0])
    _play(policy, forward=True, jaw=-0.17)
    _play(policy, forward=False, jaw=1.74)
    assert float(policy._hold_action[0]) == 10.0, "UP must end back at the hover pose"


def test_failed_insert_plan_still_round_trips():
    # A one-waypoint trajectory; replaying it backwards must not walk off the end.
    policy = _policy_with_traj([10.0])
    assert _play(policy, forward=True, jaw=-0.17) == [10.0]
    assert _play(policy, forward=False, jaw=1.74) == [10.0]


if __name__ == "__main__":
    for name, case in sorted(globals().items()):
        if name.startswith("test_"):
            case()
            print(f"ok {name}")
    print("all retreat playback checks passed")
