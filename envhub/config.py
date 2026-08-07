"""LeRobot env configs for this repo's Arena environments.

These subclass LeRobot's ``IsaaclabArenaEnv``, which already knows how to turn Arena
observations into policy inputs (``IsaaclabArenaProcessorStep``: the ``state_keys``
terms concatenated into ``observation.state``, one ``observation.images.<key>`` per
camera key). The only behavioural change is clearing ``hub_path``: with no hub path,
``lerobot.envs.make_env`` takes the local ``create_envs`` branch instead of
downloading an ``env.py`` from the Hugging Face Hub.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym

from lerobot.envs.configs import EnvConfig, IsaaclabArenaEnv

from .env import make_env


@EnvConfig.register_subclass("shape_sorting_arena")
@dataclass
class ShapeSortingArenaEnv(IsaaclabArenaEnv):
    """SO-101 shape sorting, configured to match the recorded training datasets."""

    hub_path: str | None = None
    """Empty so ``make_env`` builds the env locally (no ``--trust_remote_code``)."""

    environment: str | None = "shape_sorting_test"
    embodiment: str | None = "so101_abs_joint"
    object: str | None = None
    """Unused: shape sorting generates its own pieces instead of picking a registry object."""
    task: str | None = None
    """Language instruction; ``None`` keeps the Arena task's own description."""

    # Observation layout — must match the dataset the policy was trained on.
    # ``joint_pos`` alone (not ``joint_vel``/``actions``) is what the recorder stored.
    state_keys: str = "joint_pos"
    camera_keys: str | None = "camera_ego_rgb,external_camera_rgb"
    state_dim: int = 6
    action_dim: int = 6
    camera_height: int = 480
    camera_width: int = 640
    enable_cameras: bool = True

    episode_length: int | None = None
    """Max rollout steps; ``None`` uses the Arena task's own ``episode_length_s``."""
    visualizer: str | None = None
    """Isaac Lab ``--viz`` backends, e.g. ``kit`` to watch the rollout in a window."""
    enable_pinocchio: bool = False
    """SO-101 uses differential IK, so Pinocchio (Pink IK) is not needed."""

    def create_envs(self, n_envs: int, use_async_envs: bool = False) -> dict[str, dict[int, gym.vector.VectorEnv]]:
        """Build the environment locally (called by ``lerobot.envs.make_env``)."""
        return make_env(n_envs=n_envs, use_async_envs=use_async_envs, cfg=self)
