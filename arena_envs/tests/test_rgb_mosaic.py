"""``envhub._rgb_mosaic`` turns camera observations into LeRobot's eval-video frames.

Regression guard for the black eval videos: the wrapper used to hand LeRobot a
``np.zeros`` placeholder whenever Isaac Lab's ``render()`` returned ``None``, which after
Isaac Lab 3.0 dropped ``rgb_array`` was *always*. Every eval video came out solid black.
Imports no ``isaaclab``, so this runs without Isaac Sim.
"""

from __future__ import annotations

import numpy as np
import torch

from envhub.isaaclab_env_wrapper import _rgb_mosaic


def _camera(num_envs: int, value: int) -> torch.Tensor:
    """A uint8 camera batch whose per-env content differs, so misrouting is visible."""
    frames = torch.zeros((num_envs, 4, 6, 3), dtype=torch.uint8)
    for env_idx in range(num_envs):
        frames[env_idx] = value + env_idx
    return frames


def test_cameras_are_laid_side_by_side_per_env():
    mosaic = _rgb_mosaic({"ego": _camera(3, 10), "external": _camera(3, 100)}, num_envs=3)

    assert mosaic.shape == (3, 4, 12, 3), mosaic.shape  # widths 6 + 6 concatenated
    assert mosaic.dtype == np.uint8
    assert mosaic.any(), "frames are black — the placeholder bug is back"
    for env_idx in range(3):
        assert (mosaic[env_idx, :, :6] == 10 + env_idx).all()
        assert (mosaic[env_idx, :, 6:] == 100 + env_idx).all()


def test_normalized_float_cameras_are_rescaled_to_uint8():
    mosaic = _rgb_mosaic({"ego": torch.full((1, 2, 2, 3), 0.5)}, num_envs=1)

    assert mosaic.dtype == np.uint8
    assert (mosaic == 127).all(), mosaic[0, 0, 0]


def test_non_rgb_modalities_are_skipped():
    depth = torch.ones((2, 4, 6, 1))
    mosaic = _rgb_mosaic({"ego": _camera(2, 10), "ego_depth": depth}, num_envs=2)

    assert mosaic.shape == (2, 4, 6, 3), "depth was encoded as RGB"


def test_no_camera_falls_back_to_a_stable_shape():
    assert _rgb_mosaic({}, num_envs=2).shape == (2, 480, 640, 3)
    assert _rgb_mosaic(None, num_envs=1).shape == (1, 480, 640, 3)


if __name__ == "__main__":
    test_cameras_are_laid_side_by_side_per_env()
    test_normalized_float_cameras_are_rescaled_to_uint8()
    test_non_rgb_modalities_are_skipped()
    test_no_camera_falls_back_to_a_stable_shape()
    print("ok")
