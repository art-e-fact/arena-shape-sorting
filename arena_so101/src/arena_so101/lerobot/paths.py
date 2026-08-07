"""Absolute paths to SO-101 LeRobot / GR00T conversion assets in this package."""

from __future__ import annotations

from pathlib import Path

PACKAGE_LEROBOT_DIR = Path(__file__).resolve().parent

MODALITY_JSON = PACKAGE_LEROBOT_DIR / "modality.json"
INFO_JSON = PACKAGE_LEROBOT_DIR / "info.json"
SIM_JOINTS_YAML = PACKAGE_LEROBOT_DIR / "6dof_joint_space.yaml"
POLICY_JOINTS_YAML = PACKAGE_LEROBOT_DIR / "gr00t_6dof_joint_space.yaml"
SHAPE_SORTING_CONFIG_YAML = PACKAGE_LEROBOT_DIR / "shape_sorting_config.yaml"
