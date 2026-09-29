# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

from shape_sorting.shape_forms import DEFAULT_EDGE_CHAMFER, DEFAULT_FORMS, DEFAULT_HOLE_CHAMFER, ShapeForm

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


PHYSICS_HZ = 200.0
"""Physics rate the scene was tuned at (Arena default: ``sim.dt=1/200``, ``decimation=4``).

``decimation`` is derived from :attr:`ShapeSortingEnvironmentCfg.control_hz` to hold the
physics rate near this value, so changing the control rate does not change contact
behaviour (grasp reliability, piece settling).
"""


def _apply_control_rate(env_cfg, control_hz: float) -> None:
    """Set the env-step rate, which is also the recorded LeRobot dataset fps.

    ``generate_policy_demos`` derives the dataset fps from ``1 / env.step_dt`` and
    ``step_dt = sim.dt * decimation``, so this is the only place the recording rate is set.

    ``render_interval`` is pinned to ``decimation`` so exactly one RTX render lands on the
    last physics substep of each env step. The counter it is checked against is never reset,
    so the alignment holds for the whole run. Anything smaller re-renders frames no
    observation ever reads (the Arena default renders twice per step).
    """
    if control_hz <= 0.0:
        raise ValueError(f"control_hz must be > 0, got {control_hz}")
    decimation = max(1, round(PHYSICS_HZ / control_hz))
    env_cfg.decimation = decimation
    env_cfg.sim.dt = 1.0 / (control_hz * decimation)
    env_cfg.sim.render_interval = decimation
    step_hz = 1.0 / (env_cfg.sim.dt * decimation)
    assert abs(step_hz - control_hz) < 1e-6, f"step rate {step_hz} != control_hz {control_hz}"


def drop_piece_on_hole(
    env,
    env_ids,
    prob: float,
    box_name: str,
    holes: dict[str, tuple[float, float, float, float, float]],
    xy_m: float = 0.010,
    yaw_rad: float = math.radians(30.0),
    tilt_rad: tuple[float, float] = (0.0, math.radians(15.0)),
    gap_m: float = 0.005,
) -> None:
    """Reset event: with ``prob``, start one random piece falling onto its own lid hole.

    Registered after Arena's placement event, so the box already stands where this episode
    has it. ``holes`` maps each piece to its hole centre in the box frame and its own size,
    ``(hole_x, hole_y, lid_top_z, half_height, radius)``. The chosen piece is posed up to
    ``xy_m`` off the hole axis, up to ``yaw_rad`` off the hole's orientation, and tipped by
    ``tilt_rad`` about a random horizontal axis, with its lowest corner ``gap_m`` above the
    lid. Its table spot is simply left empty.

    Calibration knobs, all of them: they set the mix of "fell in", "wedged" and "on the
    lid", which is only observable from the policy's logs. A bad release lets go of a
    piece that is upright in the jaws, and the rim is what tips it — so the offset does the
    work and the tilt stays small. Live, spawning pieces tipped 20-40° instead often
    toppled them onto their side (the round one rolls off the rim) — a different state,
    which :func:`tip_piece_over` produces on purpose.
    """
    import torch
    import warp as wp
    from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul

    if env_ids is None or len(env_ids) == 0 or prob <= 0.0:
        return
    box_pose = wp.to_torch(env.scene[box_name].data.root_pose_w)  # (x, y, z, qx, qy, qz, qw)
    device = box_pose.device
    for env_id in env_ids.tolist():
        if float(torch.rand(())) >= prob:
            continue
        box_pos, box_quat = box_pose[env_id, :3], box_pose[env_id, 3:7]
        name = list(holes)[int(torch.randint(len(holes), ()))]
        hx, hy, lid_z, half_height, radius = holes[name]
        u = torch.rand(5).tolist()
        dx, dy = (2 * u[0] - 1) * xy_m, (2 * u[1] - 1) * xy_m
        yaw = (2 * u[2] - 1) * yaw_rad
        tilt = tilt_rad[0] + u[3] * (tilt_rad[1] - tilt_rad[0])
        axis_yaw = (2 * u[4] - 1) * math.pi
        q_yaw = quat_from_angle_axis(torch.tensor([yaw]), torch.tensor([[0.0, 0.0, 1.0]]))
        q_tilt = quat_from_angle_axis(
            torch.tensor([tilt]), torch.tensor([[math.cos(axis_yaw), math.sin(axis_yaw), 0.0]])
        )
        quat = quat_mul(q_tilt, quat_mul(q_yaw, box_quat.cpu().unsqueeze(0)))[0].to(device)
        hole = box_pos + quat_apply(box_quat, torch.tensor([hx, hy, lid_z], device=device))
        # Lowest corner of a tipped prism: h·cos t + R·sin t below its origin.
        lift = gap_m + half_height * math.cos(tilt) + radius * math.sin(tilt)
        pos = hole + torch.tensor([dx, dy, lift], device=device)
        ids = torch.tensor([env_id], device=device)
        piece = env.scene[name]
        piece.write_root_pose_to_sim(torch.cat([pos, quat]).unsqueeze(0), env_ids=ids)
        piece.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=ids)
        print(
            f"[drop_piece_on_hole] env {env_id}: {name} dropped on its hole — "
            f"tilt {math.degrees(tilt):.0f}°, yaw {math.degrees(yaw):+.0f}°, "
            f"off ({dx * 1000:+.0f}, {dy * 1000:+.0f}) mm"
        )


def tip_piece_over(
    env,
    env_ids,
    prob: float,
    pieces: dict[str, tuple[float, float]],
    gap_m: float = 0.005,
) -> None:
    """Reset event: with ``prob``, lay one random piece on its side where it stands.

    ``pieces`` maps each candidate to ``(half_height, radius)``. The piece is turned 90°
    about a random horizontal axis and dropped from ``gap_m`` above the table, so it
    settles on whichever side face is nearest — the state a knocked-over piece is in.
    """
    import torch
    import warp as wp
    from isaaclab.utils.math import quat_from_angle_axis, quat_mul

    if env_ids is None or len(env_ids) == 0 or prob <= 0.0 or not pieces:
        return
    for env_id in env_ids.tolist():
        if float(torch.rand(())) >= prob:
            continue
        name = list(pieces)[int(torch.randint(len(pieces), ()))]
        half_height, radius = pieces[name]
        piece = env.scene[name]
        pose = wp.to_torch(piece.data.root_pose_w)[env_id].clone()
        axis_yaw = (2 * float(torch.rand(())) - 1) * math.pi
        q_tip = quat_from_angle_axis(
            torch.tensor([math.pi / 2]), torch.tensor([[math.cos(axis_yaw), math.sin(axis_yaw), 0.0]])
        )
        pose[3:7] = quat_mul(q_tip, pose[3:7].cpu().unsqueeze(0))[0].to(pose.device)
        # Standing, its bottom was half_height below the origin; lying, no point of it is
        # more than radius below.
        pose[2] += radius - half_height + gap_m
        ids = torch.tensor([env_id], device=pose.device)
        piece.write_root_pose_to_sim(pose.unsqueeze(0), env_ids=ids)
        piece.write_root_velocity_to_sim(torch.zeros(1, 6, device=pose.device), env_ids=ids)
        print(f"[tip_piece_over] env {env_id}: {name} laid on its side.")


@dataclass(frozen=True)
class ShapeInfo:
    """Privileged per-piece metadata attached to the manager env cfg.

    Consumed by policies that need layout knowledge beyond the ``policy`` obs
    group (e.g. cuRobo). Safe for other policies to ignore.
    """

    prim_path: str
    """USD prim path expression (may include ``{ENV_REGEX_NS}``)."""

    @property
    def name(self) -> str:
        """Scene entity name — last segment of :attr:`prim_path`."""
        return self.prim_path.rstrip("/").rsplit("/", 1)[-1]


@dataclass
class ShapeSortingEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the shape-sorting pick-and-place environment."""

    enable_cameras: bool = False
    embodiment: str = "droid_rel_joint_pos"
    teleop_device: str | None = None
    leader_port: str = "/dev/ttyACM0"
    leader_id: str = "leader"
    leader_recalibrate: bool = False
    hdr: str | None = None
    light_intensity: float = 500.0
    additional_table_objects: list[str] = field(default_factory=list)
    rl_training_mode: bool = False
    forms: list[ShapeForm] = field(default_factory=lambda: list(DEFAULT_FORMS))
    piece_size: float = 0.03
    piece_height: float = 0.03
    box_height: float = 0.04
    box_mass: float = 0.35
    """Sorting box mass [kg]. At 0.35 kg (friction 0.6) about 2 N slides it, and pulling a
    piece that is wedged in a hole out of it does: 42-45 mm of box drift per run with
    ``drop_on_hole_prob`` on, against 4-7 mm without — enough that later inserts miss."""
    clearance: float = 0.003
    edge_chamfer: float = DEFAULT_EDGE_CHAMFER
    hole_chamfer: float = DEFAULT_HOLE_CHAMFER
    debug_key_reset: bool = False
    """If True, press R (Isaac window focused) to end the current episode early."""
    episode_length_s: float = 40.0
    """Episode timeout in seconds. Failed rollouts truncate when this elapses."""
    control_hz: float = 30.0
    """Env-step rate [Hz], and therefore the fps of datasets recorded from this env.

    Physics stays near :data:`PHYSICS_HZ`; only the action/observation/render rate changes.
    Use an integer — LeRobot rejects a non-integer fps. Note this does not by itself change
    how many frames an episode has: scripted phases are step-counted, so lowering the rate
    stretches the demo in wall-clock time instead. Pair it with the CuroboPolicy pacing
    flags (``--waypoint_stride``, ``--close_steps``, ``--open_steps``, ``--home_steps``)
    to keep the arm's real-world speed."""
    drop_on_hole_prob: float = 0.0
    """Probability per episode that one random piece starts dropped onto its own lid hole.

    The piece is let fall from just above the hole, off-centre, yawed and tilted, and
    physics decides the rest: it drops in, wedges in the hole, or lands on the lid. The
    last two are what a trained policy leaves behind when it lets go in the wrong place —
    a state the scripted policy's own inserts never produce, because it checks alignment
    before letting go. One piece, not each: that is the state being recreated, and live,
    three pieces on the lid at once left the arm — which cannot lift much above the lid
    over the box — no collision-free way to carry any of them out. For dataset
    generation; keep it 0 for evaluation."""
    tip_over_prob: float = 0.0
    """Probability per episode that one random piece starts lying on its side.

    A knocked-over piece is the other state a trained policy leaves behind, and one
    the scripted policy must stand back up before it fits its hole. Never the cube when
    it is as tall as it is wide: on its side it is the same cube, standing. Applied
    before ``drop_on_hole_prob``, so the two never fight over one piece. For dataset
    generation; keep it 0 for evaluation."""
    reset_robot_joint_noise: float = 0.0
    """Uniform ±noise [rad] around the robot's default joint pose at every episode reset.

    Arena registers no robot reset event, so without this term the arm keeps whatever pose
    the previous episode ended in — typically tangled with the freshly re-placed sorting box.
    0 resets it exactly to the default pose; a small value also varies the start state so a
    learned policy sees more than one initial configuration."""


class ShapeSortingEnvironment(ArenaEnvironmentFactory[ShapeSortingEnvironmentCfg]):
    """Registered provider for the procedural shape-sorting environment."""

    name: str = "shape_sorting_test"
    _legacy_argparse_cfg_type = ShapeSortingEnvironmentCfg

    def build(self, cfg: ShapeSortingEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab.envs.common import ViewerCfg
        from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg

        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor, On, PositionLimits
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose

        from shape_sorting.debug_key_reset import debug_key_reset_termination
        from shape_sorting.shape_asset import SortingBox, make_shape_sorting_layout
        from shape_sorting.sorting_task import ShapeSortingTask

        # SO-101 embodiments and devices (so101_*, including so101_gamepad) live in arena_so101.
        if cfg.embodiment.startswith("so101") or (cfg.teleop_device or "").startswith("so101"):
            import arena_so101

            if not hasattr(arena_so101, "register"):
                raise ImportError(
                    "import arena_so101 did not load the package (got a shadowed path). "
                    "Install arena-so101 from GitHub (see arena_envs/pyproject.toml) "
                    "or set ARENA_SO101_PATH to a local checkout (see DEVELOPMENT.md)."
                )
            arena_so101.register()

        # Retrieve assets from the registry / build the sorting layout.
        background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        layout = make_shape_sorting_layout(
            forms=cfg.forms,
            piece_size=cfg.piece_size,
            piece_height=cfg.piece_height,
            box_height=cfg.box_height,
            box_mass=cfg.box_mass,
            clearance=cfg.clearance,
            edge_chamfer=cfg.edge_chamfer,
            hole_chamfer=cfg.hole_chamfer,
        )

        # Describe spatial relationships.
        table_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
        table_reference.add_relation(IsAnchor())

        # +X points in front of the robot on the table.
        # +Y points to the left of the robot on the table.
        layout.box.add_relation(On(table_reference))
        layout.box.add_relation(PositionLimits(x_min=0.4, x_max=0.41, y_min=-0.101, y_max=-0.1))

        for asset in layout.pieces:
            asset.add_relation(On(table_reference))
            asset.add_relation(PositionLimits(x_min=0.4, x_max=0.5, y_min=-0.01, y_max=0.2))

        additional_table_objects = [
            self.asset_registry.get_asset_by_name(name)() for name in cfg.additional_table_objects
        ]
        for obj in additional_table_objects:
            obj.add_relation(On(table_reference))

        # Configure lighting.
        light = self.asset_registry.get_asset_by_name("light")()
        light.set_intensity(cfg.light_intensity)
        if cfg.hdr is not None:
            light.add_hdr(self.hdr_registry.get_hdr_by_name(cfg.hdr)())
        directional_light = self.asset_registry.get_asset_by_name("directional_light")()

        # Select the embodiment (flat obs vector for RL policy input).
        embodiment_kwargs = dict(enable_cameras=cfg.enable_cameras, concatenate_observation_terms=True)
        if "so101" in cfg.embodiment:
            # Sit on the side of the table, facing +X (arena_so101 composes the base yaw).
            embodiment_kwargs["initial_pose"] = Pose(position_xyz=(0.236, 0.0, -0.027))
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(**embodiment_kwargs)

        if "so101" in cfg.embodiment:
            # Over-shoulder third-person view of the sorting table: eye and target relative to
            # the robot base (env frame (1.2, -0.55, 0.9) -> (0.45, 0.0, 0.05)). The camera is
            # attached to the base link, so it moves with the robot.
            if cfg.enable_cameras and embodiment.camera_config is not None:
                embodiment.set_external_camera_view((0.964, -0.55, 0.927), (0.214, 0.0, 0.077))
            # Back to the initial joint pose (± noise) on every reset: arena_so101 owns the
            # event, this only sets the noise. Not collision-checked; keep it small.
            embodiment.event_config.reset_robot_joints.params["position_range"] = (
                -cfg.reset_robot_joint_noise,
                cfg.reset_robot_joint_noise,
            )

        # Droid does not wire concatenate_observation_terms into its obs cfg yet.
        embodiment.observation_config.policy.concatenate_terms = True

        teleop_device = None
        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
            if cfg.teleop_device == "so101_leader":
                teleop_device.port = cfg.leader_port
                teleop_device.leader_id = cfg.leader_id
                teleop_device.leader_recalibrate = cfg.leader_recalibrate

        # Step 5: Compose the scene.
        scene = Scene(
            assets=[
                background,
                light,
                directional_light,
                table_reference,
                *layout.assets(),
                *additional_table_objects,
            ]
        )

        # Place-all task: success when every piece center is inside the box cavity.
        cavity = layout.box.get_inner_bounding_box()
        task = ShapeSortingTask(
            pick_up_object_list=layout.pieces,
            destination_location_list=[layout.box] * len(layout.pieces),
            background_scene=background,
            episode_length_s=cfg.episode_length_s,
            piece_in_box_params={
                "object_cfg_list": [SceneEntityCfg(piece.name) for piece in layout.pieces],
                "container_cfg": SceneEntityCfg(layout.box.name),
                "aabb_min": tuple(cavity.min_point[0].tolist()),
                "aabb_max": tuple(cavity.max_point[0].tolist()),
                "velocity_threshold": 0.1,
            },
        )
        # SortMultiObjectTask takes no description. Must match the recorded datasets'
        # task text: language-conditioned policies (SmolVLA) tokenize it every step.
        task.task_description = "Insert the shapes into the sorting box."

        # Privileged lid-hole frames (box-local offsets) for policies / debug viz.
        from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg

        from isaaclab_arena.utils.configclass import combine_configclass_instances, make_configclass

        # TODO: debug_vis should be a parameter in the environment cfg.
        hole_frames_cfg = layout.box.get_hole_frames_cfg(debug_vis=False)
        HoleFramesSceneCfg = make_configclass(
            "HoleFramesSceneCfg",
            [(SortingBox.HOLE_FRAMES_SENSOR_NAME, FrameTransformerCfg, hole_frames_cfg)],
        )
        task.scene_config = combine_configclass_instances(
            "SceneCfg",
            task.scene_config,
            HoleFramesSceneCfg(),
        )

        # Privileged layout metadata + viewer / debug hooks for the manager env cfg.
        def _configure_env_cfg(env_cfg):
            _apply_control_rate(env_cfg, cfg.control_hz)
            env_cfg.viewer = ViewerCfg(eye=(1.5, 0.0, 1.0), lookat=(0.2, 0.0, 0.0))
            env_cfg.shapes = [ShapeInfo(prim_path=piece.prim_path) for piece in layout.pieces]
            # Same params as the success predicate, so a policy can verify one piece with the
            # exact success criterion by overriding ``object_cfg_list``, and recording scripts
            # can poll ``pieces_in_box`` after nulling ``terminations.success``.
            env_cfg.piece_in_box_params = task.piece_in_box_params
            # Both added here, after the placement event composed into env_cfg.events, and
            # events run in the order they were added: box and pieces have their episode
            # poses. Half-height and plan radius per piece, from its own bounding box.
            sizes = {}
            for piece in layout.pieces:
                bbox = piece.get_bounding_box()
                half = (bbox.max_point[0] - bbox.min_point[0]) / 2.0
                sizes[piece.name] = (float(half[2]), float(math.hypot(half[0], half[1])))
            if cfg.tip_over_prob > 0.0:
                tippable = {
                    name: size for name, size in sizes.items()
                    if not (name.endswith(ShapeForm.CUBE.value) and cfg.piece_height == cfg.piece_size)
                }
                env_cfg.events.tip_piece_over = EventTermCfg(
                    func=tip_piece_over,
                    mode="reset",
                    params={"prob": cfg.tip_over_prob, "pieces": tippable},
                )
            if cfg.drop_on_hole_prob > 0.0:
                holes = {
                    piece.name: (float(hx), float(hy), float(layout.box.lid_top_z), *sizes[piece.name])
                    for piece, (hx, hy) in zip(layout.pieces, layout.box.hole_centers)
                }
                env_cfg.events.drop_piece_on_hole = EventTermCfg(
                    func=drop_piece_on_hole,
                    mode="reset",
                    params={"prob": cfg.drop_on_hole_prob, "box_name": layout.box.name, "holes": holes},
                )
            if cfg.debug_key_reset:
                # Truncation (not success): ends the episode so policy_runner resets and continues.
                env_cfg.terminations.debug_key_reset = TerminationTermCfg(
                    func=debug_key_reset_termination,
                    time_out=True,
                )
            return env_cfg

        # Assemble the environment.
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=_configure_env_cfg,
        )
        return isaaclab_arena_environment
