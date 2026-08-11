"""Adapt a batched Isaac Lab environment to the vector-env API LeRobot expects.

Isaac Lab already runs ``num_envs`` sub-environments inside one GPU simulation, so this
is a thin translation layer (tensors to numpy, success flags into ``info``) rather than a
real vectorizer. It subclasses ``AsyncVectorEnv`` — without calling its ``__init__``,
there are no worker processes — because LeRobot branches on that type to reach
sub-environment state through ``call()`` instead of indexing ``env.envs``.
"""

from __future__ import annotations

import atexit
import logging
from contextlib import suppress
from typing import Any

import gymnasium as gym
import numpy as np
import torch

logger = logging.getLogger(__name__)


def close_simulation(env, simulation_app) -> None:
    """Close the Isaac Lab env and the simulation app, ignoring teardown errors."""
    with suppress(Exception):
        if env is not None:
            env.close()
    with suppress(Exception):
        if simulation_app is not None:
            simulation_app.app.close()


class IsaacLabEnvWrapper(gym.vector.AsyncVectorEnv):
    """Expose one batched Isaac Lab env as a LeRobot-compatible vector env."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        env: gym.Env,
        *,
        episode_length: int,
        task: str | None = None,
        simulation_app: Any = None,
    ):
        self._env = env
        self._episode_length = episode_length
        self._simulation_app = simulation_app
        self._closed = False
        self.task = task

        self.render_mode = env.render_mode
        self.observation_space = self.single_observation_space = env.observation_space
        self.action_space = self.single_action_space = env.action_space
        # LeRobot times its eval videos with render_fps, which Isaac Lab's metadata omits.
        self.metadata = {
            **self.metadata,
            **(env.metadata or {}),
            "render_fps": round(1.0 / env.unwrapped.step_dt),
        }

        atexit.register(self.close)

    @property
    def unwrapped(self) -> IsaacLabEnvWrapper:
        return self

    @property
    def num_envs(self) -> int:
        return self._env.unwrapped.num_envs

    @property
    def device(self) -> str:
        return self._env.unwrapped.device

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # Vector envs may pass one seed per sub-env; Isaac Lab seeds the whole simulation once.
        if isinstance(seed, (list, tuple, range)):
            seed = seed[0] if len(seed) > 0 else None

        obs, info = self._env.reset(seed=seed, options=options)
        info["final_info"] = {"is_success": np.zeros(self.num_envs, dtype=bool)}
        return obs, info

    def step(self, actions: np.ndarray | torch.Tensor) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict]:
        obs, reward, terminated, truncated, info = self._env.step(torch.as_tensor(actions, device=self.device))

        reward = reward.cpu().numpy().astype(np.float32)
        terminated = terminated.cpu().numpy().astype(bool)
        truncated = truncated.cpu().numpy().astype(bool)
        # LeRobot reads per-env success from info["final_info"]["is_success"].
        info["final_info"] = {"is_success": self._success_flags(terminated | truncated)}

        return obs, reward, terminated, truncated, info

    def _success_flags(self, done: np.ndarray) -> np.ndarray:
        """Per-env success taken from Arena's ``success`` termination term, on done steps only."""
        terminations = self._env.unwrapped.termination_manager
        if "success" not in terminations.active_terms:
            return np.zeros(self.num_envs, dtype=bool)
        return terminations.get_term("success").cpu().numpy().astype(bool) & done

    def call(self, name: str, *args, **kwargs) -> list[Any]:
        """Answer LeRobot's per-sub-environment queries with one entry per env."""
        if name == "_max_episode_steps":
            return [self._episode_length] * self.num_envs
        if name in ("task", "task_description"):
            return [self.task] * self.num_envs
        if name == "render":
            frame = self.render()
            # LeRobot stacks these into a video, so keep the shape stable without a viewport.
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            return [frame] * self.num_envs
        raise AttributeError(f"IsaacLabEnvWrapper does not expose '{name}'")

    def get_attr(self, name: str) -> list[Any]:
        return self.call(name)

    def render(self) -> np.ndarray | None:
        """Return a single RGB viewport frame (LeRobot uses these for its eval videos)."""
        if self.render_mode != "rgb_array":
            return None
        frame = self._env.render()
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        if frame is None:
            return None
        # Isaac Lab may return a batch of frames; the viewport shows only one.
        return frame[0] if frame.ndim == 4 else frame

    def close(self, **kwargs) -> None:
        """Close the environment and shut down the simulation app (once)."""
        if self._closed:
            return
        self._closed = True
        logger.info("Closing Isaac Lab Arena environment")
        close_simulation(self._env, self._simulation_app)

    def __enter__(self) -> IsaacLabEnvWrapper:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False
