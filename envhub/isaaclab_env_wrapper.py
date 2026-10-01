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
    """Close the Isaac Lab env and the simulation app, ignoring teardown errors.

    Prefer this for failure paths where the process is about to abort anyway.
    For normal LeRobot eval shutdown, :meth:`IsaacLabEnvWrapper.close` only closes
    the env and defers ``app.close()`` to atexit — Kit's ``app.close()`` can
    hard-exit the process and skip writing ``eval_info.json``.
    """
    with suppress(Exception):
        if env is not None:
            env.close()
    _close_simulation_app(simulation_app)


def _rgb_mosaic(camera_obs: dict[str, Any] | None, num_envs: int) -> np.ndarray:
    """Lay one step's RGB camera observations side by side: ``(num_envs, H, W * cameras, 3)`` uint8.

    These stand in for the viewport LeRobot wants for its eval videos. Isaac Lab 3.0 dropped
    ``render_mode="rgb_array"`` — ``ManagerBasedRLEnv.render()`` warns and returns ``None`` for
    every mode — so there is no viewport frame to hand over. The camera observations are the
    policy's own inputs, already rendered every step, so they cost nothing extra and show what
    the policy actually saw, which is what a failure needs to be read from.
    """
    # Cameras arrive as (num_envs, H, W, 3); depth and segmentation carry a different channel
    # count and are skipped rather than encoded as bogus RGB.
    frames = [f for f in (camera_obs or {}).values() if f.shape[-1] == 3]
    if not frames:
        # No RGB camera configured, so there is genuinely nothing to show; LeRobot stacks
        # whatever it gets and needs a stable shape.
        return np.zeros((num_envs, 480, 640, 3), dtype=np.uint8)
    mosaic = torch.cat([torch.as_tensor(f) for f in frames], dim=2).cpu().numpy()
    # mdp.image(normalize=True) yields float in [0, 1]; this repo's cameras are uint8 already.
    return mosaic if mosaic.dtype == np.uint8 else np.clip(mosaic * 255.0, 0, 255).astype(np.uint8)


def _close_simulation_app(simulation_app) -> None:
    """Shut down Kit. May hard-exit the process; flush stdio first."""
    if simulation_app is None:
        return
    # Kit's app.close() can terminate with exit code 0 before Python unwinds.
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    with suppress(Exception):
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
        # Last step's camera observations, for render(). LeRobot's render callback runs straight
        # after reset() and each step(), so holding the tensors (not a copy) is enough.
        self._camera_obs: dict[str, Any] | None = None

        # Staged progress diagnostics. Binary success runs at a few percent, too sparse to tell two
        # policies apart at any affordable episode count, so also score how far each piece got.
        # This env defines no reward terms, so the reward channel is free — and LeRobot already
        # writes per-episode ``max_rewards`` into eval_info.json, which is exactly the latched
        # best progress. Absent on envs that do not publish the geometry; reward stays as it was.
        self._stage_params = getattr(env.unwrapped.cfg, "piece_stage_params", None)
        self._stages: torch.Tensor | None = None
        self._rest_z: torch.Tensor | None = None

        self.render_mode = env.render_mode
        self.observation_space = self.single_observation_space = env.observation_space
        self.action_space = self.single_action_space = env.action_space
        # LeRobot times its eval videos with render_fps, which Isaac Lab's metadata omits.
        self.metadata = {
            **self.metadata,
            **(env.metadata or {}),
            "render_fps": round(1.0 / env.unwrapped.step_dt),
        }

        # close() runs from LeRobot's close_envs *before* eval_info.json is written.
        # Defer Kit shutdown to atexit so that write can complete.
        atexit.register(self._atexit_close)

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
        self._camera_obs = obs.get("camera_obs")
        self._stages = self._rest_z = None
        info["final_info"] = {"is_success": np.zeros(self.num_envs, dtype=bool)}
        return obs, info

    def step(self, actions: np.ndarray | torch.Tensor) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict]:
        obs, reward, terminated, truncated, info = self._env.step(torch.as_tensor(actions, device=self.device))
        self._camera_obs = obs.get("camera_obs")

        reward = reward.cpu().numpy().astype(np.float32)
        terminated = terminated.cpu().numpy().astype(bool)
        truncated = truncated.cpu().numpy().astype(bool)
        if self._stage_params is not None:
            reward = self._update_progress(terminated, truncated)
        # LeRobot reads per-env success from info["final_info"]["is_success"].
        info["final_info"] = {"is_success": self._success_flags(terminated | truncated)}

        return obs, reward, terminated, truncated, info

    def _update_progress(self, terminated: np.ndarray, truncated: np.ndarray) -> np.ndarray:
        """Latch each piece's best stage this episode and return progress in [0, 1] per env.

        Monotone within an episode, so LeRobot's per-episode ``max_rewards`` is the final score.
        Prints one line per finished episode with the per-piece stages, which ``eval_checkpoints.sh``
        aggregates into a histogram — the mean alone cannot say *which* stage the policy stalls at.
        """
        from shape_sorting.predicates import MAX_STAGE, STAGE_NAMES, piece_stages, read_piece_poses

        local_xyz, world_z = read_piece_poses(
            self._env, self._stage_params["object_names"], self._stage_params["container_name"]
        )
        device = world_z.device
        done_t = torch.as_tensor(terminated | truncated, device=device)

        if self._stages is None:
            self._stages = torch.zeros(world_z.shape, dtype=torch.int8, device=device)
            self._rest_z = world_z.clone()
        self._rest_z = torch.minimum(self._rest_z, world_z)

        stage = piece_stages(local_xyz, world_z, self._rest_z, **self._stage_params["geometry"])
        # Isaac Lab auto-resets a done env inside step(), so the scene state visible now for those
        # envs is already the *next* episode's — drop this step's reading for them.
        stage = torch.where(done_t.unsqueeze(-1), torch.zeros_like(stage), stage)
        self._stages = torch.maximum(self._stages, stage)
        # That reset would otherwise lose the very step an insertion completes, so credit the final
        # stage outright on success. Not on `terminated`: Arena routes failures like object_dropped
        # through it too, which would score a piece knocked off the table as a full box.
        succeeded = self._success_flags(terminated | truncated)
        self._stages[torch.as_tensor(succeeded, device=device)] = MAX_STAGE

        progress = self._stages.float().mean(dim=1) / MAX_STAGE
        names = self._stage_params["object_names"]
        for env_idx in np.flatnonzero(terminated | truncated):
            stages = self._stages[env_idx].tolist()
            # piece_stages stays machine-readable for eval_checkpoints.sh; the named form is for
            # reading the log, where which *piece* stalls matters as much as which stage.
            print(
                f"[stages] progress={float(progress[env_idx]):.3f} "
                f"piece_stages={','.join(str(s) for s in stages)} "
                + " ".join(f"{name}={STAGE_NAMES[stage]}" for name, stage in zip(names, stages)),
                flush=True,
            )
        # Clear the finished episodes; the next step's reading starts their successors.
        self._stages[done_t] = 0
        self._rest_z[done_t] = world_z[done_t]
        return progress.cpu().numpy().astype(np.float32)

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
            # One frame per sub-env: LeRobot maps them onto the episodes running concurrently.
            return list(_rgb_mosaic(self._camera_obs, self.num_envs))
        raise AttributeError(f"IsaacLabEnvWrapper does not expose '{name}'")

    def get_attr(self, name: str) -> list[Any]:
        return self.call(name)

    def render(self) -> np.ndarray:
        """One RGB frame for sub-env 0 (``call('render')`` serves the whole batch)."""
        return _rgb_mosaic(self._camera_obs, self.num_envs)[0]

    def close(self, **kwargs) -> None:
        """Close the Isaac Lab env only (once).

        Does not call Kit ``app.close()`` here: LeRobot's ``lerobot-eval`` closes
        envs *before* writing ``eval_info.json``, and Kit can hard-exit on
        ``app.close()``. The app is shut down from :meth:`_atexit_close` instead.
        """
        if self._closed:
            return
        self._closed = True
        logger.info("Closing Isaac Lab Arena environment")
        with suppress(Exception):
            if self._env is not None:
                self._env.close()

    def _atexit_close(self) -> None:
        """Final teardown: env (if still open) then Kit app."""
        self.close()
        app = self._simulation_app
        self._simulation_app = None
        _close_simulation_app(app)

    def __enter__(self) -> IsaacLabEnvWrapper:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False
