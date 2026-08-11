# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Record successful LeRobot v3 demos by rolling out a scripted Arena policy.

Fork of Arena ``policy_runner.py`` with success filtering and a
``generate_dataset.py``-style stop condition:

* ``--generation_num_trials`` — target number of successful demos
* ``--max_retries`` — stop after this many failed episodes (optional)
* ``--output_dir`` — local LeRobot v3 dataset root

Only successful episodes are committed; failed episodes are discarded.

Scripted policies may implement ``is_demonstration_ended() -> bool`` so a finished
sequence ends the episode early instead of waiting for timeout (failed attempt if
task success has not held for ``--num_success_steps``).

Example::

    python -m shape_sorting.generate_policy_demos \\
      --viz kit \\
      --policy_type shape_sorting.curobo_policy.CuroboPolicy \\
      --generation_num_trials 10 \\
      --max_retries 40 \\
      --action_noise 0.01 \\
      --output_dir ./datasets/curobo_shape_sorting \\
      --dataset_repo_id local/curobo_shape_sorting \\
      --debug_viser \\
      shape_sorting_test \\
      --embodiment so101_abs_joint
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.policy_runner import get_policy_cls
from isaaclab_arena.evaluation.policy_runner_cli import (
    add_policy_cli_args,
    build_policy_from_cli,
)
from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena_environments.cli import (
    get_arena_builder_from_cli,
    get_isaaclab_arena_environments_cli_parser,
)

if TYPE_CHECKING:
    import gymnasium as gym
    from arena_so101.lerobot.recorder import SO101LeRobotRecorder
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
        "--output_dir",
        type=str,
        default="./datasets/curobo_shape_sorting",
        help="Local root directory for the LeRobot v3 dataset.",
    )
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        default="local/curobo_shape_sorting",
        help="LeRobot dataset identifier stored in metadata.",
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing compatible dataset in --output_dir.",
    )
    output_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate an existing --output_dir.",
    )
    parser.add_argument(
        "--disable_streaming_encoding",
        action="store_true",
        help="Write temporary PNGs and encode videos at episode end instead of streaming.",
    )
    parser.add_argument(
        "--task_description",
        type=str,
        default=None,
        help="Dataset task text. Defaults to the environment language instruction.",
    )
    parser.add_argument(
        "--num_success_steps",
        type=int,
        default=10,
        help="Consecutive steps with task success required before saving a demo.",
    )
    parser.add_argument(
        "--disable_full_sim_buffer_reset",
        action="store_true",
        default=False,
        help="Skip env.sim.reset() before each new recording episode.",
    )
    parser.add_argument(
        "--action_noise",
        type=float,
        default=0.0,
        help=(
            "Std-dev of i.i.d. Gaussian noise added to every action dim (including jaw) "
            "before env.step, Mimic-style. Recorded actions are the noisy ones. "
            "0 disables noise. Typical Mimic scales are ~0.003–0.03 depending on action space."
        ),
    )


def _configure_env_for_recording(env_cfg: Any) -> Any:
    """Disable success termination and return the term for manual polling."""
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        raise NotImplementedError(
            "No success termination term was found in the environment. "
            "Cannot filter LeRobot episodes by success."
        )

    # Keep time_out so failed rollouts truncate and free the retry budget.
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.recorders = None

    return success_term


def _get_processed_actions(base_env: gym.Env):
    """Return the concatenated action targets applied during the latest step."""
    import torch

    processed_actions = [
        base_env.action_manager.get_term(term_name).processed_actions
        for term_name in base_env.action_manager.active_terms
    ]
    assert processed_actions, "The environment has no active action terms"
    return torch.cat(processed_actions, dim=-1)


def _apply_action_noise(actions, action_noise: float):
    """Add i.i.d. Gaussian noise to actions (Mimic-style execution noise).

    Noise is applied to all action dimensions, including the jaw for continuous
    abs-joint control. Returns ``actions`` unchanged when ``action_noise <= 0``.
    """
    import torch

    if action_noise <= 0.0:
        return actions
    return actions + action_noise * torch.randn_like(actions)


def _reset_recording_episode(
    env: gym.Env,
    policy: PolicyBase,
    *,
    disable_full_sim_buffer_reset: bool,
    task_description: str,
) -> Any:
    base_env = env.unwrapped
    if not disable_full_sim_buffer_reset:
        base_env.sim.reset()
    obs, _ = env.reset()
    policy.reset()
    policy.set_task_description(task_description)
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
    recorder: SO101LeRobotRecorder,
    success_term: TerminationTermCfg,
    *,
    generation_num_trials: int,
    max_retries: int | None,
    num_success_steps: int,
    disable_full_sim_buffer_reset: bool,
    task_description: str | None,
    action_noise: float,
    is_running,
) -> tuple[int, int]:
    """Roll out ``policy`` until enough successes or the retry budget is exhausted.

    ``env`` should be the gym-wrapped env from ``gym.make`` / ``make_registered``.

    When ``action_noise > 0``, i.i.d. Gaussian noise is added to policy actions before
    ``env.step`` (including jaw). Recorded actions come from the env's processed
    (noisy) targets, matching Mimic datagen.

    Returns:
        (num_successful, num_failed)
    """
    import torch

    base_env = env.unwrapped
    assert base_env.num_envs == 1, (
        "generate_policy_demos currently requires --num_envs 1"
    )
    if action_noise < 0.0:
        raise ValueError(f"--action_noise must be >= 0, got {action_noise}")
    task_description = task_description or base_env.get_language_instruction()
    if not task_description:
        raise ValueError("A task description is required; pass --task_description")

    num_successful = 0
    num_failed = 0
    success_step_count = 0

    obs = _reset_recording_episode(
        env,
        policy,
        disable_full_sim_buffer_reset=disable_full_sim_buffer_reset,
        task_description=task_description,
    )

    print(
        f"Collecting demos: target={generation_num_trials} successes"
        + (
            f", max_retries={max_retries}"
            if max_retries is not None
            else ", max_retries=unlimited"
        )
        + (f", action_noise={action_noise}" if action_noise > 0.0 else ", action_noise=off")
    )

    with torch.inference_mode():
        while is_running():
            if num_successful >= generation_num_trials:
                break
            if max_retries is not None and num_failed >= max_retries:
                print(f"Reached max_retries={max_retries} failed episodes; stopping.")
                break

            actions = _apply_action_noise(policy.get_action(env, obs), action_noise)
            recording_observation = recorder.snapshot_observation(obs)
            obs, _, terminated, truncated, _ = env.step(actions)
            recorder.add_transition(
                recording_observation,
                _get_processed_actions(base_env),
                task=task_description,
            )

            if _is_success(base_env, success_term):
                success_step_count += 1
                if success_step_count >= num_success_steps:
                    recorder.save_episode()
                    num_successful += 1
                    print(
                        f"Saved successful demo {num_successful}/{generation_num_trials} "
                        f"(failed so far: {num_failed})."
                    )
                    success_step_count = 0
                    if num_successful >= generation_num_trials:
                        break
                    obs = _reset_recording_episode(
                        env,
                        policy,
                        disable_full_sim_buffer_reset=disable_full_sim_buffer_reset,
                        task_description=task_description,
                    )
                    continue
            else:
                success_step_count = 0

            # Timeout / other truncation or non-success termination → failed attempt.
            # Manager-based envs auto-reset on done; obs is already the next episode.
            if terminated.any() or truncated.any():
                recorder.discard_episode()
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
                recorder.discard_episode()
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
                    task_description=task_description,
                )

    return num_successful, num_failed


def main() -> None:
    # Script-local registration only — do not decorate the factory with
    # @register_environment (that breaks Arena scripts using --external_environment_class_path).
    _register_shape_sorting_environment()

    args_parser = get_isaaclab_arena_cli_parser()
    _add_generation_arguments(args_parser)
    args_cli, _ = args_parser.parse_known_args()

    # Cameras must be enabled before Kit starts for the LeRobot video features.
    args_cli.enable_cameras = True

    with SimulationAppContext(args_cli) as sim_app:
        # Resolve policy class, then attach its typed CLI flags + env subparsers.
        policy_cls = get_policy_cls(args_cli.policy_type)
        print(f"Policy type: {args_cli.policy_type} -> {policy_cls}")

        args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
        args_parser = add_policy_cli_args(args_parser, policy_cls)
        args_cli, hydra_overrides = args_parser.parse_known_args()
        assert_hydra_overrides(hydra_overrides, args_parser)
        # Full re-parse resets enable_cameras to its default.
        args_cli.enable_cameras = True

        if args_cli.num_envs != 1:
            raise ValueError(
                f"generate_policy_demos requires --num_envs 1, got {args_cli.num_envs}"
            )

        arena_builder = get_arena_builder_from_cli(
            args_cli, hydra_overrides=hydra_overrides
        )
        env_name, env_cfg, env_kwargs = arena_builder.build_registered()
        success_term = _configure_env_for_recording(env_cfg)

        import gymnasium as gym
        from arena_so101.lerobot.recorder import SO101LeRobotRecorder
        from isaaclab_arena.utils.isaaclab_utils.simulation_app import (
            reapply_viewer_cfg,
        )

        env = gym.make(env_name, cfg=env_cfg, **env_kwargs)
        reapply_viewer_cfg(env)
        policy = build_policy_from_cli(policy_cls, args_cli)

        try:
            fps_float = 1.0 / env.unwrapped.step_dt
            fps = round(fps_float)
            if not math.isclose(fps_float, fps, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(f"LeRobot requires an integer FPS, got {fps_float}")

            with SO101LeRobotRecorder(
                root=args_cli.output_dir,
                repo_id=args_cli.dataset_repo_id,
                fps=fps,
                resume=args_cli.resume,
                overwrite=args_cli.overwrite,
                streaming_encoding=not args_cli.disable_streaming_encoding,
            ) as recorder:
                print(
                    f"LeRobot dataset: {recorder.root} "
                    f"(existing episodes: {recorder.num_episodes}, fps: {fps})"
                )
                num_successful, num_failed = collect_policy_demos(
                    env,
                    policy,
                    recorder,
                    success_term,
                    generation_num_trials=args_cli.generation_num_trials,
                    max_retries=args_cli.max_retries,
                    num_success_steps=args_cli.num_success_steps,
                    disable_full_sim_buffer_reset=args_cli.disable_full_sim_buffer_reset,
                    task_description=args_cli.task_description,
                    action_noise=args_cli.action_noise,
                    is_running=sim_app.is_running,
                )
        finally:
            if policy.is_remote:
                policy.shutdown_remote(
                    kill_server=getattr(args_cli, "remote_kill_on_exit", False)
                )
            env.close()

        print(
            f"Done: saved {num_successful} successful demo(s), "
            f"{num_failed} failed attempt(s) → {args_cli.output_dir}"
        )


if __name__ == "__main__":
    main()
