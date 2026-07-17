"""Map LeRobot SO-101 leader readings ↔ workshop USD joint radians.

LeRobot uses names like ``shoulder_pan.pos``; the workshop USD uses
``Rotation``, ``Pitch``, … ``Jaw``. Order is identical — we map by index.
"""

from __future__ import annotations

import math

import torch

# Sim USD joint order (must match ArticulationCfg / action cfg).
SIM_JOINT_NAMES = (
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
    "Jaw",
)

# LeRobot leader keys in the same order.
LEROBOT_JOINT_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

# USD joint limits in degrees (from NVIDIA SO-101 workshop).
_USD_LIMITS_DEG = (
    (-110.0, 110.0),  # shoulder_pan / Rotation
    (-100.0, 100.0),  # shoulder_lift / Pitch
    (-100.0, 90.0),  # elbow_flex / Elbow
    (-95.0, 95.0),  # wrist_flex / Wrist_Pitch
    (-160.0, 160.0),  # wrist_roll / Wrist_Roll
    (-10.0, 100.0),  # gripper / Jaw
)


def leader_dict_to_sim_radians(
    leader_action: dict[str, float],
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert a LeRobot leader action dict to a (6,) sim joint-position tensor."""
    raw = torch.tensor(
        [float(leader_action[k]) for k in LEROBOT_JOINT_KEYS],
        dtype=torch.float32,
        device=device,
    )
    return motor_norm_to_sim_radians(raw)


def motor_norm_to_sim_radians(raw_values: torch.Tensor) -> torch.Tensor:
    """Map LeRobot motor-norm degrees → sim radians.

    Arm joints are reported in [-100, 100]; gripper in [0, 100].
    """
    mins = torch.tensor([lo for lo, _ in _USD_LIMITS_DEG], dtype=raw_values.dtype, device=raw_values.device)
    maxs = torch.tensor([hi for _, hi in _USD_LIMITS_DEG], dtype=raw_values.dtype, device=raw_values.device)

    normalized = torch.zeros_like(raw_values)
    normalized[:-1] = (raw_values[:-1] + 100.0) / 200.0
    normalized[-1] = raw_values[-1] / 100.0

    mapped_deg = mins + normalized * (maxs - mins)
    return mapped_deg * (math.pi / 180.0)


def sim_radians_to_motor_norm(sim_radians: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`motor_norm_to_sim_radians` (for dataset export)."""
    mins = torch.tensor([lo for lo, _ in _USD_LIMITS_DEG], dtype=sim_radians.dtype, device=sim_radians.device)
    maxs = torch.tensor([hi for _, hi in _USD_LIMITS_DEG], dtype=sim_radians.dtype, device=sim_radians.device)

    mapped_deg = sim_radians * (180.0 / math.pi)
    normalized = (mapped_deg - mins) / (maxs - mins)

    raw = torch.zeros_like(normalized)
    raw[:-1] = normalized[:-1] * 200.0 - 100.0
    raw[-1] = normalized[-1] * 100.0
    return raw
