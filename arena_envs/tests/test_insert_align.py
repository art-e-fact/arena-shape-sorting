# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Checks the piece-vs-hole alignment test that triggers an insert recovery.

Run: python test_insert_align.py — needs torch + arena_so101 (the Arena venv), no Isaac Sim.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shape_sorting.curobo_policy import (
    _PHASE_STEPS,
    CuroboPolicy,
    CuroboPolicyCfg,
    Phase,
    alignment_error,
)
from shape_sorting.shape_sorting_env import drop_piece_on_hole

_ORIGIN = torch.tensor([0.4, -0.1, 0.07])
_IDENTITY = torch.tensor([0.0, 0.0, 0.0, 1.0])


def _yaw_quat(deg: float) -> torch.Tensor:
    half = math.radians(deg) * 0.5
    return torch.tensor([0.0, 0.0, math.sin(half), math.cos(half)])


def _tilt_quat(deg: float) -> torch.Tensor:
    """Rotation about base X — tips the piece over sideways."""
    half = math.radians(deg) * 0.5
    return torch.tensor([math.sin(half), 0.0, 0.0, math.cos(half)])


def _err(obj_pos=None, obj_quat=None, hole_quat=None, order=4):
    return alignment_error(
        _ORIGIN if obj_pos is None else obj_pos,
        _IDENTITY if obj_quat is None else obj_quat,
        _ORIGIN,
        _IDENTITY if hole_quat is None else hole_quat,
        symmetry_order=order,
    )


def test_seated_piece_reports_no_error():
    xy, yaw, tilt = _err()
    assert xy < 1e-6 and yaw < 1e-6 and tilt < 1e-6, (xy, yaw, tilt)


def test_xy_offset_is_the_planar_distance():
    xy, _, _ = _err(obj_pos=_ORIGIN + torch.tensor([0.008, 0.006, 0.0]))
    assert abs(xy - 0.01) < 1e-6, xy
    # Height alone is not a misalignment: the piece hangs above the hole by design.
    xy, _, _ = _err(obj_pos=_ORIGIN + torch.tensor([0.0, 0.0, 0.05]))
    assert xy < 1e-6, xy


def test_yaw_folds_into_the_piece_symmetry():
    # A square turned by a quarter turn is the same square.
    assert _err(obj_quat=_yaw_quat(90.0))[1] < 1e-5
    assert abs(math.degrees(_err(obj_quat=_yaw_quat(30.0))[1]) - 30.0) < 1e-3
    assert abs(math.degrees(_err(obj_quat=_yaw_quat(100.0))[1]) - 10.0) < 1e-3
    # Sign does not matter; the recovery only needs the magnitude.
    assert abs(math.degrees(_err(obj_quat=_yaw_quat(-30.0))[1]) - 30.0) < 1e-3


def test_yaw_is_measured_against_the_hole_not_the_world():
    _, yaw, _ = _err(obj_quat=_yaw_quat(45.0), hole_quat=_yaw_quat(45.0))
    assert yaw < 1e-5, yaw


def test_cylinder_never_has_a_yaw_error():
    assert _err(obj_quat=_yaw_quat(37.0), order=None)[1] == 0.0


def test_tilt_is_the_angle_off_vertical():
    assert abs(math.degrees(_err(obj_quat=_tilt_quat(20.0))[2]) - 20.0) < 1e-3
    # Lying flat / upside down must stay inside acos's domain, not blow up.
    assert abs(math.degrees(_err(obj_quat=_tilt_quat(90.0))[2]) - 90.0) < 1e-3
    assert abs(math.degrees(_err(obj_quat=_tilt_quat(180.0))[2]) - 180.0) < 1e-3


def test_yaw_tolerance_stays_under_the_geometric_bind_angle():
    # A square of side s in a hole inflated by clearance c binds once
    # s*(cos+sin) > s + 2c: 13.1 deg at the default 30 mm piece / 3 mm clearance.
    # A looser tolerance would let the policy commit to placements that cannot drop in.
    assert CuroboPolicyCfg().insert_align_yaw_tol_rad < math.radians(13.1)


def test_every_phase_has_a_step_handler():
    # A phase with no handler does not raise — get_action just holds the last action, so
    # the arm freezes for the rest of the episode and the rollout times out silently.
    assert set(_PHASE_STEPS) | {Phase.DONE} == set(Phase)
    assert Phase.DONE not in _PHASE_STEPS, "DONE holds; giving it a handler would loop"
    for phase, name in _PHASE_STEPS.items():
        assert callable(getattr(CuroboPolicy, name, None)), f"{phase} -> missing {name}"


def test_tilt_budget_is_generous_enough_to_try_the_insert():
    # A slightly tipped piece still drops in, and no re-aim can level it, so a tight
    # tolerance only buys pointless parking. Must stay above the yaw tolerance, which
    # is the error that *can* be re-aimed away.
    cfg = CuroboPolicyCfg()
    assert cfg.insert_align_tilt_tol_rad >= math.radians(15.0)
    assert cfg.insert_align_tilt_tol_rad > cfg.insert_align_yaw_tol_rad


def test_a_piece_dropped_on_its_hole_is_measured_against_that_hole():
    # The env's drop event and the policy's alignment check must agree on what "lined up
    # with the hole" means, or a dropped piece is misjudged the moment the policy looks.
    import warp as wp

    wp.init()

    box_pos, box_quat = torch.tensor([0.4, -0.1, 0.05]), _yaw_quat(90.0)  # turned: offsets turn too
    written = {}
    piece = SimpleNamespace(
        write_root_pose_to_sim=lambda pose, env_ids: written.update(pose=pose[0]),
        write_root_velocity_to_sim=lambda vel, env_ids: None,
    )
    box = SimpleNamespace(data=SimpleNamespace(root_pose_w=wp.from_torch(torch.cat([box_pos, box_quat])[None])))
    env = SimpleNamespace(scene={"sorting_box": box, "shape_piece_cube": piece})
    hole = box_pos + torch.tensor([-0.01, 0.03, 0.02])  # (0.03, 0.01, lid 0.02) turned 90°

    def drop(tilt):
        drop_piece_on_hole(
            env, torch.tensor([0]), prob=1.0, box_name="sorting_box",
            holes={"shape_piece_cube": (0.03, 0.01, 0.02, 0.015, 0.021)},
            xy_m=0.0, yaw_rad=0.0, tilt_rad=(tilt, tilt), gap_m=0.005,
        )
        pose = written["pose"]
        return pose, alignment_error(pose[:3], pose[3:], hole, box_quat, symmetry_order=4)

    pose, (xy, yaw, tilt) = drop(0.0)
    assert xy < 1e-6 and yaw < 1e-5 and tilt < 1e-3, (xy, yaw, tilt)
    assert abs(float(pose[2]) - (float(hole[2]) + 0.005 + 0.015)) < 1e-6, "bottom 5 mm over the lid"
    pose, (xy, _, tilt) = drop(math.radians(30.0))
    assert xy < 1e-6 and abs(math.degrees(tilt) - 30.0) < 1e-3, (xy, tilt)


if __name__ == "__main__":
    for name, case in sorted(globals().items()):
        if name.startswith("test_"):
            case()
            print(f"ok {name}")
    print("all insert alignment checks passed")
