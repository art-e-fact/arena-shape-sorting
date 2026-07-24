# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Segmented demo recording: plan freely, then replay/smooth-execute with checkpoints.

Phases
  PLANNING  — teleop without recording; checkpoint saved on entry
  EXECUTING — reset to checkpoint, play path, record, then back to PLANNING

Keys (U I O P cluster + Backspace / R)
  I          reset to current checkpoint (retry this segment)
  U          undo previous committed segment
  P          execute replay of the planned actions
  O          execute smooth lerp from first→last planned action
  Backspace  abort episode
  R          hard reset (same as abort + env.reset)

Works with any teleop/action space. Smooth lerp is best for absolute actions
(e.g. SO-101 joints); delta SE3 users should prefer replay (P).
"""

from __future__ import annotations

import contextlib
import copy
import os
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--dataset_file", type=str, required=True, help="HDF5 path for exported demos.")
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument("--num_demos", type=int, default=1, help="Stop after N successful demos (0 = infinite).")
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Consecutive success steps required before exporting a demo.",
)
parser.add_argument(
    "--smooth_steps",
    type=int,
    default=30,
    help="Number of steps for exec-smooth interpolation.",
)
parser.add_argument(
    "--disable_full_sim_buffer_reset",
    action="store_true",
    default=False,
    help="Skip env.sim.reset() on hard episode reset.",
)
add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)
if "openxr" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import gymnasium as gym
import omni
import omni.log
import torch

import isaaclab_mimic.envs  # noqa: F401
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
from isaaclab.utils.datasets import EpisodeData
from isaaclab_teleop import IsaacTeleopCfg, create_isaac_teleop_device, remove_camera_configs

from isaaclab_arena.utils.isaaclab_utils.recorders import ArenaEnvRecorderManagerCfg
from isaaclab_arena.utils.isaaclab_utils.simulation_app import reapply_viewer_cfg

HELP = "[PLANNING] I=retry U=undo P=replay O=smooth Backspace=abort R=reset"


class Phase(Enum):
    PLANNING = auto()
    EXECUTING = auto()


@dataclass
class Segment:
    """One committed plan→execute chunk."""

    ckpt_start: dict
    episode: EpisodeData  # full episode snapshot after this segment


@dataclass
class Session:
    phase: Phase = Phase.PLANNING
    ckpt_current: dict | None = None
    plan_actions: list[torch.Tensor] = field(default_factory=list)
    stack: list[Segment] = field(default_factory=list)
    exec_queue: list[torch.Tensor] = field(default_factory=list)
    pending: str | None = None  # retry | undo | replay | smooth | abort | reset
    recording: bool = False


class RateLimiter:
    def __init__(self, hz: int):
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)
        self.last_time = time.time()

    def sleep(self, env):
        next_wakeup = self.last_time + self.sleep_duration
        while time.time() < next_wakeup:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time += self.sleep_duration
        while self.last_time < time.time():
            self.last_time += self.sleep_duration


class MetaKeyboard:
    """Phase-machine keys, independent of the active teleop device."""

    def __init__(self, callbacks: dict[str, Callable[[], None]]):
        self._callbacks = callbacks
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_event(event),
        )

    def _on_event(self, event) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input.name in self._callbacks:
            self._callbacks[event.input.name]()
        return True

    def shutdown(self):
        if self._sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            self._sub = None


def clone_state(state: dict) -> dict:
    return copy.deepcopy(state)


def clone_episode(episode: EpisodeData) -> EpisodeData:
    cloned = EpisodeData()
    cloned.data = copy.deepcopy(episode.data)
    cloned.env_id = episode.env_id
    cloned.success = episode.success
    return cloned


def set_episode(env, episode: EpisodeData) -> None:
    env.recorder_manager._episodes[0] = episode


def gate_recorder(env, enabled: bool) -> None:
    """Skip pre/post step recording while planning."""
    rm = env.recorder_manager
    if not hasattr(rm, "_ungated_pre_step"):
        rm._ungated_pre_step = rm.record_pre_step
        rm._ungated_post_step = rm.record_post_step

    def _pre():
        if enabled:
            rm._ungated_pre_step()

    def _post():
        if enabled:
            rm._ungated_post_step()

    rm.record_pre_step = _pre  # type: ignore[method-assign]
    rm.record_post_step = _post  # type: ignore[method-assign]


def restore_scene(env, ckpt: dict) -> None:
    """Kinematic rewind without wiping the recorder episode."""
    env.scene.reset_to(ckpt, env_ids=None, is_relative=True)
    env.sim.forward()


def make_smooth_actions(plan_actions: list[torch.Tensor], num_steps: int) -> list[torch.Tensor]:
    """Lerp first→last planned action. Best for absolute action spaces."""
    if not plan_actions:
        return []
    if len(plan_actions) == 1 or num_steps <= 1:
        return [plan_actions[-1].clone()]
    a0, a1 = plan_actions[0], plan_actions[-1]
    steps = max(2, num_steps)
    return [a0 + (a1 - a0) * (i / (steps - 1)) for i in range(steps)]


def status(session: Session, extra: str = "") -> None:
    msg = f"{HELP} | segments={len(session.stack)}"
    if extra:
        msg = f"{msg} | {extra}"
    print(msg)


def begin_planning(env, session: Session, note: str = "") -> None:
    session.phase = Phase.PLANNING
    session.ckpt_current = clone_state(env.scene.get_state(is_relative=True))
    session.plan_actions.clear()
    session.exec_queue.clear()
    session.recording = False
    gate_recorder(env, False)
    status(session, note or "PLANNING")


def hard_reset_episode(env, session: Session, teleop) -> None:
    if not args_cli.disable_full_sim_buffer_reset:
        env.sim.reset()
    env.recorder_manager.reset()
    env.reset()
    teleop.reset()
    session.stack.clear()
    session.pending = None
    begin_planning(env, session, "episode reset")


def undo_segment(env, session: Session) -> None:
    if not session.stack:
        print("Nothing to undo.")
        return
    seg = session.stack.pop()
    restore_scene(env, seg.ckpt_start)
    if session.stack:
        set_episode(env, clone_episode(session.stack[-1].episode))
    else:
        env.recorder_manager.reset()
        env.recorder_manager.record_post_reset([0])
    begin_planning(env, session, "undone segment")


def start_execute(env, session: Session, mode: str) -> None:
    if not session.plan_actions:
        print("No planned actions — move first, then P/O.")
        return
    if session.ckpt_current is None:
        print("No checkpoint — aborting execute.")
        return

    if mode == "replay":
        session.exec_queue = [a.clone() for a in session.plan_actions]
    else:
        session.exec_queue = make_smooth_actions(session.plan_actions, args_cli.smooth_steps)

    # Rewind to segment start, keep committed episode intact, then record the path.
    restore_scene(env, session.ckpt_current)
    if session.stack:
        set_episode(env, clone_episode(session.stack[-1].episode))
    else:
        env.recorder_manager.reset()
        env.recorder_manager.record_post_reset([0])

    session.phase = Phase.EXECUTING
    session.recording = True
    gate_recorder(env, True)
    status(session, f"EXECUTING ({mode}, {len(session.exec_queue)} steps)")


def finish_execute(env, session: Session) -> None:
    assert session.ckpt_current is not None
    session.stack.append(
        Segment(ckpt_start=session.ckpt_current, episode=clone_episode(env.recorder_manager.get_episode(0)))
    )
    begin_planning(env, session, "segment committed")


def try_export_after_execute(env, session: Session, success_term) -> bool:
    """Check success only after an execute path finishes. Hold-records until confirmed.

    Returns True if the demo was exported.
    """
    if success_term is None:
        return False

    hold = session.plan_actions[-1].to(env.device) if session.plan_actions else None
    for i in range(args_cli.num_success_steps):
        if not bool(success_term.func(env, **success_term.params)[0]):
            return False
        # Record still frames between checks so success is consecutive in the demo.
        if i + 1 < args_cli.num_success_steps and hold is not None:
            env.step(hold.repeat(env.num_envs, 1))

    export_success(env)
    return True


def export_success(env) -> None:
    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
    env.recorder_manager.set_success_to_episodes([0], torch.tensor([[True]], dtype=torch.bool, device=env.device))
    env.recorder_manager.export_episodes([0])
    print("Success — demo exported.")


def setup_env():
    output_dir = os.path.dirname(args_cli.dataset_file) or "."
    output_file = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    os.makedirs(output_dir, exist_ok=True)

    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg, env_kwargs = arena_builder.build_registered()

    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        omni.log.warn("No success termination — demos cannot be auto-marked successful.")

    if args_cli.xr:
        if not args_cli.enable_cameras:
            env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.recorders = ArenaEnvRecorderManagerCfg() if args_cli.enable_cameras else ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    # Mid-episode reset_to / restore must not auto-export.
    env_cfg.recorders.export_in_record_pre_reset = False

    env = gym.make(env_name, cfg=env_cfg, **env_kwargs).unwrapped
    reapply_viewer_cfg(env)
    return env, env_cfg, success_term


def setup_teleop(env_cfg):
    callbacks = {}
    if hasattr(env_cfg, "isaac_teleop") and isinstance(env_cfg.isaac_teleop, IsaacTeleopCfg):
        return create_isaac_teleop_device(env_cfg.isaac_teleop, sim_device=env_cfg.sim.device, callbacks=callbacks)

    if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
        device_cfg = env_cfg.teleop_devices.devices[args_cli.teleop_device]
        print(f"[teleop] Creating {args_cli.teleop_device!r} from {type(device_cfg).__name__}")
        return create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks)

    name = args_cli.teleop_device.lower()
    if name == "keyboard":
        return Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
    if name == "spacemouse":
        return Se3SpaceMouse(Se3SpaceMouseCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
    raise RuntimeError(
        f"Unsupported teleop device: {args_cli.teleop_device!r} "
        f"(teleop_devices keys: {list(getattr(getattr(env_cfg, 'teleop_devices', None), 'devices', {}) or {})})"
    )


def run(env, env_cfg, success_term, rate_limiter: RateLimiter | None) -> int:
    session = Session()
    teleop = setup_teleop(env_cfg)

    def set_pending(cmd: str):
        def _cb():
            # Ignore plan commands while a path is playing out.
            if session.phase == Phase.EXECUTING and cmd not in ("abort", "reset"):
                return
            session.pending = cmd

        return _cb

    meta = MetaKeyboard(
        {
            "I": set_pending("retry"),
            "U": set_pending("undo"),
            "P": set_pending("replay"),
            "O": set_pending("smooth"),
            "BACKSPACE": set_pending("abort"),
            "R": set_pending("reset"),
        }
    )

    demos = 0
    use_isaac_teleop = hasattr(teleop, "__enter__") and hasattr(teleop, "__exit__")

    def loop():
        nonlocal demos
        hard_reset_episode(env, session, teleop)
        print("Segmented recording started.")
        print(HELP)

        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while simulation_app.is_running():
                cmd = session.pending
                session.pending = None

                if cmd == "reset" or cmd == "abort":
                    hard_reset_episode(env, session, teleop)
                elif session.phase == Phase.PLANNING:
                    if cmd == "retry" and session.ckpt_current is not None:
                        restore_scene(env, session.ckpt_current)
                        session.plan_actions.clear()
                        status(session, "retry current checkpoint")
                    elif cmd == "undo":
                        undo_segment(env, session)
                    elif cmd in ("replay", "smooth"):
                        start_execute(env, session, cmd)

                action = teleop.advance()
                if action is None:
                    env.sim.render()
                    if rate_limiter:
                        rate_limiter.sleep(env)
                    continue

                if session.phase == Phase.PLANNING:
                    env.step(action.repeat(env.num_envs, 1))
                    session.plan_actions.append(action.detach().cpu().clone())

                elif session.phase == Phase.EXECUTING:
                    if session.exec_queue:
                        step_action = session.exec_queue.pop(0).to(env.device)
                        env.step(step_action.repeat(env.num_envs, 1))
                    if not session.exec_queue:
                        # Success is evaluated only at the end of execute, never while planning.
                        if try_export_after_execute(env, session, success_term):
                            demos = env.recorder_manager.exported_successful_episode_count
                            print(f"Recorded {demos} successful demonstrations.")
                            if args_cli.num_demos > 0 and demos >= args_cli.num_demos:
                                break
                            hard_reset_episode(env, session, teleop)
                        else:
                            finish_execute(env, session)

                if env.sim.is_stopped():
                    break
                if rate_limiter:
                    rate_limiter.sleep(env)

    try:
        if use_isaac_teleop:
            with teleop:
                loop()
        else:
            loop()
    finally:
        meta.shutdown()

    return demos


def main() -> None:
    rate_limiter = None if args_cli.xr else RateLimiter(args_cli.step_hz)
    if args_cli.xr:
        from isaaclab.ui.xr_widgets import TeleopVisualizationManager, XRVisualization

        XRVisualization.assign_manager(TeleopVisualizationManager)

    env, env_cfg, success_term = setup_env()
    demos = run(env, env_cfg, success_term, rate_limiter)
    env.close()
    print(f"Done — {demos} demos → {args_cli.dataset_file}")


if __name__ == "__main__":
    main()
    simulation_app.close()
