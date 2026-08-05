# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Record successful imitation-learning demos by rolling out a scripted Arena policy.

Fork of Arena ``record_demos.py`` (HDF5 export / success filtering) + ``policy_runner.py``
(policy CLI / rollout), with a ``generate_dataset.py``-style stop condition:

* ``--generation_num_trials`` — target number of successful demos
* ``--max_retries`` — stop after this many failed episodes (optional)
* ``--output_file`` — HDF5 path for exported demos

Only episodes marked successful are written (``EXPORT_SUCCEEDED_ONLY``).

Scripted policies may implement ``is_demonstration_ended() -> bool`` so a finished
sequence ends the episode early instead of waiting for timeout (failed attempt if
task success has not held for ``--num_success_steps``).

Example::

    python -m shape_sorting.generate_policy_demos \\
      --viz kit \\
      --policy_type shape_sorting.curobo_policy.CuroboPolicy \\
      --generation_num_trials 10 \\
      --max_retries 40 \\
      --output_file ./datasets/curobo_shape_sorting.hdf5 \\
      --debug_viser \\
      shape_sorting_test \\
      --embodiment so101_abs_joint
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.policy_runner import get_policy_cls
from isaaclab_arena.evaluation.policy_runner_cli import add_policy_cli_args, build_policy_from_cli
from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

if TYPE_CHECKING:
    import gymnasium as gym
    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.policy.policy_base import PolicyBase


def _register_shape_sorting_environment() -> None:
    """Make ``shape_sorting_test`` available as an Arena CLI subcommand for this script only."""
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from shape_sorting.shape_sorting_env import ShapeSortingEnvironment

    registry = EnvironmentRegistry()
    if not registry.is_registered(ShapeSortingEnvironment.name, ensure_loaded=False):
        registry.register(ShapeSortingEnvironment, ShapeSortingEnvironment.name)


def _add_generation_arguments(parser) -> None:
    """CLI flags shared with generate_dataset / record_demos stop conditions."""
    parser.add_argument(
        "--policy_type",
        type=str,
        required=True,
        help="Registered policy name or dotted module.Class path (same as policy_runner).",
    )
    parser.add_argument(
        "--generation_num_trials",
        type=int,
        required=True,
        help="Target number of successful demos to export.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=None,
        help="Stop after this many failed episodes (successful demos do not count). Default: unlimited.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./datasets/policy_dataset.hdf5",
        help="HDF5 path for exported successful demos.",
    )
    parser.add_argument(
        "--num_success_steps",
        type=int,
        default=10,
        help="Consecutive steps with task success required before exporting a demo.",
    )
    parser.add_argument(
        "--disable_full_sim_buffer_reset",
        action="store_true",
        default=False,
        help="Skip env.sim.reset() before each new recording episode.",
    )


def _setup_output_paths(output_file: str) -> tuple[str, str]:
    output_dir = os.path.dirname(output_file) or "."
    output_file_name = os.path.splitext(os.path.basename(output_file))[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir, output_file_name


def _configure_env_for_recording(
    env_cfg: Any,
    args_cli: Any,
    output_dir: str,
    output_file_name: str,
) -> Any:
    """Attach IL recorders and extract the success term (polled in the rollout loop)."""
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import DatasetExportMode

    from isaaclab_arena.utils.isaaclab_utils.recorders import ArenaEnvRecorderManagerCfg

    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        raise NotImplementedError(
            "No success termination term was found in the environment. "
            "Cannot mark demos as successful for EXPORT_SUCCEEDED_ONLY."
        )

    # Keep time_out so failed rollouts truncate and free the retry budget.
    env_cfg.observations.policy.concatenate_terms = False

    if args_cli.enable_cameras:
        env_cfg.recorders = ArenaEnvRecorderManagerCfg()
    else:
        env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return success_term


def _export_successful_episode(base_env: gym.Env) -> None:
    import torch

    base_env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
    base_env.recorder_manager.set_success_to_episodes(
        [0], torch.tensor([[True]], dtype=torch.bool, device=base_env.device)
    )
    base_env.recorder_manager.export_episodes([0])


def _reset_recording_episode(
    env: gym.Env,
    policy: PolicyBase,
    *,
    disable_full_sim_buffer_reset: bool,
) -> Any:
    base_env = env.unwrapped
    if not disable_full_sim_buffer_reset:
        base_env.sim.reset()
    base_env.recorder_manager.reset()
    obs, _ = env.reset()
    policy.reset()
    policy.set_task_description(base_env.get_language_instruction())
    return obs


def _is_success(base_env: gym.Env, success_term: TerminationTermCfg) -> bool:
    return bool(success_term.func(base_env, **success_term.params)[0])


def _is_demonstration_ended(policy: PolicyBase) -> bool:
    """Duck-typed hook for scripted policies that know when their sequence is over.

    Arena ``PolicyBase`` has no such API; policies used for synthetic demos may
    implement ``is_demonstration_ended() -> bool`` (e.g. ``CuroboPolicy``).
    """
    check = getattr(policy, "is_demonstration_ended", None)
    return bool(check()) if callable(check) else False


def collect_policy_demos(
    env: gym.Env,
    policy: PolicyBase,
    success_term: TerminationTermCfg,
    *,
    generation_num_trials: int,
    max_retries: int | None,
    num_success_steps: int,
    disable_full_sim_buffer_reset: bool,
    is_running,
) -> tuple[int, int]:
    """Roll out ``policy`` until enough successes or the retry budget is exhausted.

    ``env`` should be the gym-wrapped env from ``gym.make`` / ``make_registered``.

    Returns:
        (num_successful, num_failed)
    """
    import torch

    base_env = env.unwrapped
    assert base_env.num_envs == 1, "generate_policy_demos currently requires --num_envs 1"

    num_successful = 0
    num_failed = 0
    success_step_count = 0

    obs = _reset_recording_episode(
        env, policy, disable_full_sim_buffer_reset=disable_full_sim_buffer_reset
    )

    print(
        f"Collecting demos: target={generation_num_trials} successes"
        + (f", max_retries={max_retries}" if max_retries is not None else ", max_retries=unlimited")
    )

    with torch.inference_mode():
        while is_running():
            if num_successful >= generation_num_trials:
                break
            if max_retries is not None and num_failed >= max_retries:
                print(f"Reached max_retries={max_retries} failed episodes; stopping.")
                break

            actions = policy.get_action(env, obs)
            obs, _, terminated, truncated, _ = env.step(actions)

            if _is_success(base_env, success_term):
                success_step_count += 1
                if success_step_count >= num_success_steps:
                    _export_successful_episode(base_env)
                    num_successful += 1
                    print(
                        f"Exported successful demo {num_successful}/{generation_num_trials} "
                        f"(failed so far: {num_failed})."
                    )
                    success_step_count = 0
                    if num_successful >= generation_num_trials:
                        break
                    obs = _reset_recording_episode(
                        env,
                        policy,
                        disable_full_sim_buffer_reset=disable_full_sim_buffer_reset,
                    )
                    continue
            else:
                success_step_count = 0

            # Timeout / other truncation or non-success termination → failed attempt.
            # Manager-based envs auto-reset on done; obs is already the next episode.
            if terminated.any() or truncated.any():
                num_failed += 1
                env_ids = (terminated | truncated).nonzero().flatten()
                policy.reset(env_ids=env_ids)
                success_step_count = 0
                print(
                    f"Failed episode {num_failed}"
                    + (f"/{max_retries}" if max_retries is not None else "")
                    + f" (successes: {num_successful}/{generation_num_trials})."
                )
                continue

            # Scripted policy finished without task success — end early instead of
            # waiting for episode timeout. If success is already holding, keep
            # stepping until num_success_steps exports above.
            if _is_demonstration_ended(policy) and success_step_count == 0:
                num_failed += 1
                print(
                    f"Policy demonstration ended without success — failed episode "
                    f"{num_failed}"
                    + (f"/{max_retries}" if max_retries is not None else "")
                    + f" (successes: {num_successful}/{generation_num_trials})."
                )
                obs = _reset_recording_episode(
                    env,
                    policy,
                    disable_full_sim_buffer_reset=disable_full_sim_buffer_reset,
                )

    return num_successful, num_failed


def main() -> None:
    # Script-local registration only — do not decorate the factory with
    # @register_environment (that breaks Arena scripts using --external_environment_class_path).
    _register_shape_sorting_environment()

    args_parser = get_isaaclab_arena_cli_parser()
    _add_generation_arguments(args_parser)
    args_cli, unknown = args_parser.parse_known_args()

    # Cameras must be enabled before Kit starts if the recorder needs them.
    if "--enable_cameras" in unknown or getattr(args_cli, "enable_cameras", False):
        args_cli.enable_cameras = True

    with SimulationAppContext(args_cli) as sim_app:
        # Resolve policy class, then attach its typed CLI flags + env subparsers.
        policy_cls = get_policy_cls(args_cli.policy_type)
        print(f"Policy type: {args_cli.policy_type} -> {policy_cls}")

        args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
        args_parser = add_policy_cli_args(args_parser, policy_cls)
        args_cli, hydra_overrides = args_parser.parse_known_args()
        assert_hydra_overrides(hydra_overrides, args_parser)
        # Full re-parse resets enable_cameras to its default; restore if cameras were requested.
        if "--enable_cameras" in unknown:
            args_cli.enable_cameras = True

        if args_cli.num_envs != 1:
            raise ValueError(f"generate_policy_demos requires --num_envs 1, got {args_cli.num_envs}")

        output_dir, output_file_name = _setup_output_paths(args_cli.output_file)

        arena_builder = get_arena_builder_from_cli(args_cli, hydra_overrides=hydra_overrides)
        env_name, env_cfg, env_kwargs = arena_builder.build_registered()
        success_term = _configure_env_for_recording(env_cfg, args_cli, output_dir, output_file_name)

        import gymnasium as gym

        from isaaclab_arena.utils.isaaclab_utils.simulation_app import reapply_viewer_cfg

        env = gym.make(env_name, cfg=env_cfg, **env_kwargs)
        reapply_viewer_cfg(env)
        policy = build_policy_from_cli(policy_cls, args_cli)

        try:
            num_successful, num_failed = collect_policy_demos(
                env,
                policy,
                success_term,
                generation_num_trials=args_cli.generation_num_trials,
                max_retries=args_cli.max_retries,
                num_success_steps=args_cli.num_success_steps,
                disable_full_sim_buffer_reset=args_cli.disable_full_sim_buffer_reset,
                is_running=sim_app.is_running,
            )
        finally:
            if policy.is_remote:
                policy.shutdown_remote(kill_server=getattr(args_cli, "remote_kill_on_exit", False))
            env.close()

        print(
            f"Done: exported {num_successful} successful demo(s), "
            f"{num_failed} failed attempt(s) → {args_cli.output_file}"
        )


if __name__ == "__main__":
    main()
