"""Checks the staged progress scoring used to diagnose why shape sorting fails.

Binary success (all pieces in the box) sits near a few percent, so evaluation also scores how
far each piece got. ``piece_stages`` is pure tensor math, so this needs torch only, no Isaac Sim.

Run: python test_piece_stages.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shape_sorting.predicates import MAX_STAGE, STAGE_NAMES, piece_stages

# A box centred on its own frame: 14 cm square footprint, lid top 3 cm up, two holes on the
# x axis. Piece 0 owns the hole at x = -0.04, piece 1 the one at x = +0.04.
GEOMETRY = {
    "aabb_min": (-0.05, -0.05, -0.04),
    "aabb_max": (0.05, 0.05, 0.02),
    "hole_centers_xy": [(-0.04, 0.0), (0.04, 0.0)],
    "hole_tolerances": [0.015, 0.015],
    "box_xy_min": (-0.07, -0.07),
    "box_xy_max": (0.07, 0.07),
    "lid_top_z": 0.03,
}
TABLE_Z = 0.8
PARKED = ((0.30, 0.0, -0.02), TABLE_Z, TABLE_Z)
"""A piece still at rest on the table, well clear of the box: stage 0."""

# (name, local_xyz in the box frame, world_z, rest_z, expected stage for piece 0)
SCENARIOS = [
    ("untouched", (0.30, 0.0, -0.02), TABLE_Z, TABLE_Z, 0),
    ("lifted, still away from the box", (0.30, 0.0, 0.05), TABLE_Z + 0.05, TABLE_Z, 1),
    ("over the lid but nowhere near a hole", (0.0, 0.06, 0.04), TABLE_Z + 0.10, TABLE_Z, 2),
    # The discriminator that makes this worth measuring: being over *a* hole is not progress.
    ("over the other piece's hole", (0.04, 0.0, 0.04), TABLE_Z + 0.10, TABLE_Z, 2),
    ("over its own hole", (-0.04, 0.0, 0.04), TABLE_Z + 0.10, TABLE_Z, 3),
    # Never lifted, yet inside: what drop_on_hole produces. An ordered chain would score this 0.
    ("inside the cavity without a lift", (0.0, 0.0, 0.0), TABLE_Z + 0.01, TABLE_Z, MAX_STAGE),
]


def _stages() -> torch.Tensor:
    """Run every scenario as its own env, with piece 0 under test and piece 1 parked."""
    local = torch.tensor([[case[1], PARKED[0]] for case in SCENARIOS], dtype=torch.float32)
    world_z = torch.tensor([[case[2], PARKED[1]] for case in SCENARIOS], dtype=torch.float32)
    rest_z = torch.tensor([[case[3], PARKED[2]] for case in SCENARIOS], dtype=torch.float32)
    return piece_stages(local, world_z, rest_z, **GEOMETRY)


def test_each_stage_is_scored():
    stages = _stages()
    assert stages.shape == (len(SCENARIOS), 2), stages.shape

    for row, (name, _, _, _, expected) in enumerate(SCENARIOS):
        actual = int(stages[row, 0])
        assert actual == expected, (
            f"{name}: expected {STAGE_NAMES[expected]} ({expected}), got {STAGE_NAMES[actual]} ({actual})"
        )


def test_a_parked_piece_scores_nothing():
    # Every env carries one piece left on the table; none of them may pick up credit from
    # whatever the other piece is doing.
    assert (_stages()[:, 1] == 0).all()


def test_stages_never_exceed_the_scale():
    stages = _stages()
    assert stages.min() >= 0 and stages.max() <= MAX_STAGE
    assert len(STAGE_NAMES) == MAX_STAGE + 1


if __name__ == "__main__":
    test_each_stage_is_scored()
    test_a_parked_piece_scores_nothing()
    test_stages_never_exceed_the_scale()
    print("ok")
