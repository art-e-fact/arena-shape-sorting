"""Build this repo's Isaac Lab Arena environments for LeRobot (EnvHub entry point).

``make_env`` is the function LeRobot's EnvHub expects. It is reached two ways:

* locally, through :meth:`envhub.config.ShapeSortingArenaEnv.create_envs`
  (``--env.type=shape_sorting_arena``)
* from the Hugging Face Hub, when this file is the ``env.py`` of an EnvHub repo
  (``--env.hub_path=<user>/<repo> --trust_remote_code=True``)

Isaac Sim is launched *inside* ``make_env``. Nothing at module level may import
``isaaclab``, because LeRobot imports this file while parsing CLI arguments.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import fields
from importlib import import_module
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np

try:
    from .isaaclab_env_wrapper import IsaacLabEnvWrapper, close_simulation
except ImportError:  # this file was loaded on its own, e.g. downloaded from the Hub
    from envhub.isaaclab_env_wrapper import IsaacLabEnvWrapper, close_simulation

if TYPE_CHECKING:
    from lerobot.envs.configs import IsaaclabArenaEnv

logger = logging.getLogger(__name__)

# ``--env.environment`` -> (module, environment factory, its typed config dataclass).
# Imported lazily, since importing them pulls in Isaac Lab.
ENVIRONMENTS: dict[str, tuple[str, str, str]] = {
    "shape_sorting_test": (
        "shape_sorting.shape_sorting_env",
        "ShapeSortingEnvironment",
        "ShapeSortingEnvironmentCfg",
    ),
}


def make_env(
    n_envs: int = 1,
    use_async_envs: bool = False,
    cfg: IsaaclabArenaEnv | None = None,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Launch Isaac Sim and return ``{environment_name: {0: vector_env}}`` for LeRobot.

    Args:
        n_envs: Sub-environments batched inside the one simulation (``--eval.batch_size``).
        use_async_envs: Ignored — Isaac Lab always batches on the GPU inside one process.
            LeRobot's ``--eval.use_async_envs`` defaults to True, so this is expected to
            be set on most runs; it just doesn't mean anything for Isaac Lab.
        cfg: LeRobot env config, e.g. :class:`envhub.config.ShapeSortingArenaEnv`.
    """
    assert cfg is not None, "make_env needs an env config (pass --env.type=shape_sorting_arena)"
    assert n_envs >= 1, f"n_envs must be at least 1, got {n_envs}"
    if use_async_envs:
        logger.info(
            "Ignoring --eval.use_async_envs=True: Isaac Lab batches environments inside a single "
            "simulation process regardless of this flag. Scale with --eval.batch_size instead."
        )

    state_keys = _split_keys(cfg.state_keys)
    camera_keys = _split_keys(cfg.camera_keys)
    assert not camera_keys or cfg.enable_cameras, "--env.camera_keys requires --env.enable_cameras=true"

    simulation_app = _launch_simulation_app(cfg)
    env = None
    try:
        env = _build_isaaclab_env(cfg, n_envs)
        _assert_observation_layout(env, cfg, state_keys, camera_keys)
        wrapped = IsaacLabEnvWrapper(
            env,
            episode_length=cfg.episode_length or int(env.unwrapped.max_episode_length),
            task=env.unwrapped.get_language_instruction(),
            simulation_app=simulation_app,
        )
    except Exception:
        # Kit's app.close() (inside close_simulation) can hard-kill the process with exit
        # code 0 before Python gets to print a traceback, silently swallowing the real
        # error. Log it first, and flush so it survives that.
        logger.exception("Failed to build the Isaac Lab Arena environment")
        sys.stdout.flush()
        sys.stderr.flush()
        close_simulation(env, simulation_app)
        raise
    logger.info("Built %s with %d env(s), task=%r", cfg.environment, wrapped.num_envs, wrapped.task)
    return {cfg.environment: {0: wrapped}}


def _split_keys(keys: str | None) -> tuple[str, ...]:
    """Split a comma-separated CLI list such as ``--env.camera_keys=cam_a,cam_b``."""
    return tuple(key.strip() for key in (keys or "").split(",") if key.strip())


def _launch_simulation_app(cfg: IsaaclabArenaEnv) -> Any:
    """Start Isaac Sim. Every ``isaaclab*`` import has to happen after this."""
    if cfg.enable_pinocchio:
        # Pinocchio (Pink IK embodiments) must be imported before Isaac Sim to avoid a symbol clash.
        import pinocchio  # noqa: F401

    from isaaclab.app import AppLauncher

    launcher_args: dict[str, Any] = {
        "headless": cfg.headless,
        "enable_cameras": cfg.enable_cameras,
        "device": cfg.device,
    }
    # Isaac Lab 3.0 selects GUI/visualizer backends with --viz; omitting it means headless.
    visualizer = getattr(cfg, "visualizer", None)
    if visualizer:
        launcher_args["visualizer"] = visualizer
    return AppLauncher(launcher_args)


def _build_isaaclab_env(cfg: IsaaclabArenaEnv, n_envs: int):
    """Compose the Arena environment and make the batched Isaac Lab env."""
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    assert cfg.environment in ENVIRONMENTS, (
        f"Unknown --env.environment={cfg.environment!r}. Add it to envhub.env.ENVIRONMENTS "
        f"(known: {sorted(ENVIRONMENTS)})"
    )
    module_name, factory_name, env_cfg_name = ENVIRONMENTS[cfg.environment]
    module = import_module(module_name)
    arena_env = getattr(module, factory_name)().build(_arena_env_cfg(getattr(module, env_cfg_name), cfg))

    builder_cfg = ArenaEnvBuilderCfg(
        num_envs=n_envs,
        device=cfg.device or "cuda:0",
        seed=cfg.seed if cfg.seed is not None else 0,
        disable_fabric=cfg.disable_fabric,
        mimic=cfg.mimic,
        language_instruction=cfg.task,
    )
    builder = ArenaEnvBuilder(arena_env, builder_cfg)
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    # LeRobot's IsaaclabArenaProcessorStep indexes obs["policy"] by term name
    # (e.g. joint_pos). Shape sorting enables concatenation for RL — undo that here.
    env_cfg.observations.policy.concatenate_terms = False
    # rgb_array lets LeRobot grab viewport frames for its evaluation videos.
    render_mode = "rgb_array" if cfg.enable_cameras else None
    return builder.make_registered(env_cfg=env_cfg, env_kwargs=env_kwargs, render_mode=render_mode)


def _arena_env_cfg(env_cfg_type: type, cfg: IsaaclabArenaEnv):
    """Fill an Arena environment config from same-named fields on the LeRobot config.

    Only ``embodiment``, ``enable_cameras`` and ``teleop_device`` exist on both configs.
    Anything else the environment accepts can be passed with
    ``--env.kwargs='{"piece_size": 0.035}'``, which ``IsaaclabArenaEnv`` promotes to
    attributes.
    """
    values = {field.name: getattr(cfg, field.name) for field in fields(env_cfg_type) if hasattr(cfg, field.name)}
    return env_cfg_type(**values)


def _assert_observation_layout(
    env,
    cfg: IsaaclabArenaEnv,
    state_keys: tuple[str, ...],
    camera_keys: tuple[str, ...],
) -> None:
    """Check the requested keys and dimensions against the built env, before rolling out.

    Without this, a typo in ``--env.state_keys`` or a stale ``--env.state_dim`` only
    surfaces as a shape mismatch deep inside the policy.
    """
    observations = env.unwrapped.observation_manager
    state_terms = observations.active_terms.get("policy", [])
    camera_terms = observations.active_terms.get("camera_obs", [])

    unknown = [key for key in state_keys if key not in state_terms]
    assert not unknown, f"--env.state_keys {unknown} are not in the policy group {sorted(state_terms)}"
    unknown = [key for key in camera_keys if key not in camera_terms]
    assert not unknown, f"--env.camera_keys {unknown} are not in the camera group {sorted(camera_terms)}"

    action_dim = int(env.action_space.shape[-1])
    assert cfg.action_dim == action_dim, f"--env.action_dim={cfg.action_dim} but the env takes {action_dim}"

    # group_obs_dim may be a single concatenated shape; group_obs_term_dim is always per-term.
    term_dims = dict(zip(state_terms, observations.group_obs_term_dim.get("policy", []), strict=True))
    state_dim = sum(int(np.prod(term_dims[key])) for key in state_keys)
    assert (
        cfg.state_dim == state_dim
    ), f"--env.state_dim={cfg.state_dim} but {list(state_keys)} concatenate to {state_dim}"
