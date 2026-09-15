# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

from shape_sorting.shape_forms import DEFAULT_EDGE_CHAMFER, DEFAULT_FORMS, DEFAULT_HOLE_CHAMFER, ShapeForm

if TYPE_CHECKING:
    from isaaclab.sensors import CameraCfg

    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


def _camera_offset_look_at(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    *,
    convention: Literal["opengl", "ros", "world"] = "ros",
) -> CameraCfg.OffsetCfg:
    """Build a ``CameraCfg.OffsetCfg`` that places the camera at ``eye`` looking at ``target``.

    Both points are in the camera parent frame (for ``{ENV_REGEX_NS}/external_camera``
    that is the env / world frame). ``CameraCfg`` has no prim look-at; only ``pos`` + ``rot``.
    """
    import torch
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils.math import (
        convert_camera_frame_orientation_convention,
        create_rotation_matrix_from_view,
        quat_from_matrix,
    )

    eyes = torch.tensor([eye], dtype=torch.float32)
    targets = torch.tensor([target], dtype=torch.float32)
    # world → view (OpenGL: forward -Z). Convert into the requested camera convention.
    rot_m = create_rotation_matrix_from_view(eyes, targets, up_axis="Z")
    quat_opengl = quat_from_matrix(rot_m)[0]
    quat = convert_camera_frame_orientation_convention(quat_opengl.unsqueeze(0), origin="opengl", target=convention)[0]
    return CameraCfg.OffsetCfg(
        pos=eye,
        rot=tuple(float(x) for x in quat.tolist()),
        convention=convention,
    )


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
    clearance: float = 0.003
    edge_chamfer: float = DEFAULT_EDGE_CHAMFER
    hole_chamfer: float = DEFAULT_HOLE_CHAMFER
    debug_key_reset: bool = False
    """If True, press R (Isaac window focused) to end the current episode early."""
    control_hz: float = 30.0
    """Env-step rate [Hz], and therefore the fps of datasets recorded from this env.

    Physics stays near :data:`PHYSICS_HZ`; only the action/observation/render rate changes.
    Use an integer — LeRobot rejects a non-integer fps. Note this does not by itself change
    how many frames an episode has: scripted phases are step-counted, so lowering the rate
    stretches the demo in wall-clock time instead. Pair it with the CuroboPolicy pacing
    flags (``--waypoint_stride``, ``--close_steps``, ``--open_steps``, ``--home_steps``)
    to keep the arm's real-world speed."""
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
        import isaaclab.envs.mdp as mdp_isaac_lab
        from isaaclab.envs.common import ViewerCfg
        from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg

        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor, On, PositionLimits
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.sorting_task import SortMultiObjectTask
        from isaaclab_arena.utils.pose import Pose

        from shape_sorting.debug_key_reset import debug_key_reset_termination
        from shape_sorting.predicates import objects_centers_inside_aabb
        from shape_sorting.shape_asset import SortingBox, make_shape_sorting_layout

        # SO-101 embodiments / devices (and Arena gamepad) live in arena_so101.
        if (
            cfg.embodiment.startswith("so101")
            or (cfg.teleop_device or "").startswith("so101")
            or cfg.teleop_device == "gamepad"
        ):
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
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.enable_cameras,
            concatenate_observation_terms=True,
        )

        # Set the initial pose for the SO-101 to sit on the side of the table.
        if "so101" in cfg.embodiment:
            embodiment.set_initial_pose(
                Pose(
                    position_xyz=(0.236, 0.0, -0.027),
                    rotation_xyzw=embodiment.scene_config.robot.init_state.rot,
                )
            )
            # Over-shoulder / elevated third-person cam looking at the sorting table.
            # CameraCfg has no look-at-prim API — set eye + target in env frame and
            # derive the offset quaternion (see ``_camera_offset_look_at``).
            if cfg.enable_cameras and embodiment.camera_config is not None:
                embodiment.camera_config.external_camera.offset = _camera_offset_look_at(
                    eye=(1.2, -0.55, 0.9),
                    target=(0.45, 0.0, 0.05),
                    convention="ros",
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
        task = SortMultiObjectTask(
            pick_up_object_list=layout.pieces,
            destination_location_list=[layout.box] * len(layout.pieces),
            background_scene=background,
            episode_length_s=40.0,
        )
        cavity = layout.box.get_inner_bounding_box()
        task.termination_cfg.success = TerminationTermCfg(
            func=objects_centers_inside_aabb,
            params={
                "object_cfg_list": [SceneEntityCfg(piece.name) for piece in layout.pieces],
                "container_cfg": SceneEntityCfg(layout.box.name),
                "aabb_min": tuple(cavity.min_point[0].tolist()),
                "aabb_max": tuple(cavity.max_point[0].tolist()),
                "velocity_threshold": 0.1,
            },
        )

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
            # Send the arm back to its default pose (± noise) on every reset. Nothing else
            # does: Arena's reset events only re-place the box and pieces, so the arm would
            # start the next episode where it dropped the last piece — often intersecting
            # the re-placed box, which the policy then has to fight its way out of.
            # ponytail: the sampled offsets are not collision-checked. Keep the noise small;
            # for wide start-state coverage, reject samples with the cuRobo world checker.
            env_cfg.events.reset_robot_joints = EventTermCfg(
                func=mdp_isaac_lab.reset_joints_by_offset,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "position_range": (-cfg.reset_robot_joint_noise, cfg.reset_robot_joint_noise),
                    "velocity_range": (0.0, 0.0),
                },
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
