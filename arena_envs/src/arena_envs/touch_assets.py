"""Procedural touch-target assets for arena_envs.

Registered with Arena's ``AssetRegistry`` via ``@register_asset``. Follows the
procedural-asset convention from ``isaaclab_arena.assets.object_library``
(``ProceduralCube`` / ``ProceduralTable``): subclass ``Object`` with ``usd_path=""``
and override ``_generate_rigid_cfg`` to spawn a shape config instead of a USD file.
"""

from __future__ import annotations

import math
import random

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, SceneEntityCfg

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose

from arena_envs.touchable import Touchable

# Kinematic + gravity disabled => the target hovers in place and never moves, so it
# stays a fixed floating goal the robot must reach out and touch. activate_contact_sensors
# is required so the Touchable contact sensor on this body reports contacts.
_TOUCH_SPHERE_SPAWN_CFG = sim_utils.SphereCfg(
    radius=0.05,
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.6, 0.9)),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01),
    activate_contact_sensors=True,
)


# ---------------------------------------------------------------------------
# Per-env uniform-in-ball spawn randomisation.
#
# Must be a module-level function (not a method) so Isaac Lab can serialise the
# EventTermCfg func reference as "arena_envs.touch_assets:randomize_sphere_in_ball".
# ---------------------------------------------------------------------------


def randomize_sphere_in_ball(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    center: tuple[float, float, float],
    radius: float,
) -> None:
    """Reset event: teleport the sphere to a uniform random point inside a ball.

    Sampling uses the cube-root radius trick so the density is uniform in volume
    (not concentrated at the centre).

    Args:
        env: The manager-based environment.
        env_ids: Indices of the environments being reset.
        asset_cfg: Scene entity config identifying the sphere rigid-body.
        center: (x, y, z) mean spawn position in env-local coordinates.
        radius: Ball radius in metres.
    """
    if env_ids is None or len(env_ids) == 0:
        return

    asset = env.scene[asset_cfg.name]
    cx, cy, cz = center

    for env_id in env_ids.tolist():
        # Sample a unit direction uniformly on the sphere via a 3-D normal draw.
        dx, dy, dz = random.gauss(0.0, 1.0), random.gauss(0.0, 1.0), random.gauss(0.0, 1.0)
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-8:
            dx, dy, dz = 1.0, 0.0, 0.0
            length = 1.0
        dx, dy, dz = dx / length, dy / length, dz / length

        # Scale by a uniform-in-volume radius (cube-root trick).
        r = radius * (random.random() ** (1.0 / 3.0))

        # Env-local position + world origin of this env sub-scene.
        origin = env.scene.env_origins[env_id]  # [3]
        pos_x = cx + dx * r + origin[0].item()
        pos_y = cy + dy * r + origin[1].item()
        pos_z = cz + dz * r + origin[2].item()

        position = torch.tensor([[pos_x, pos_y, pos_z]], device=env.device)
        # Identity quaternion (xyzw); kinematic sphere has no meaningful orientation.
        orientation = math_utils.quat_from_euler_xyz(
            torch.zeros(1, device=env.device),
            torch.zeros(1, device=env.device),
            torch.zeros(1, device=env.device),
        )
        root_pose = torch.cat([position, orientation], dim=-1)
        asset.write_root_pose_to_sim_index(
            root_pose=root_pose,
            env_ids=torch.tensor([env_id], device=env.device),
        )
        asset.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(1, 6, device=env.device),
            env_ids=torch.tensor([env_id], device=env.device),
        )


@register_asset
class TouchSphere(Object, Touchable):
    """A floating, kinematic procedural sphere that reports end-effector touches."""

    name = "touch_sphere"
    tags = ["object", "touchable", "procedural"]

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        touch_force_threshold: float = 1.0,
        spawn_radius: float = 0.0,
    ):
        # Set before super().__init__(): Object.__init__ calls _init_event_cfg().
        self.spawn_radius = spawn_radius
        resolved_name = instance_name if instance_name is not None else "touch_sphere"
        resolved_prim = prim_path if prim_path is not None else "{ENV_REGEX_NS}/TouchSphere"
        super().__init__(
            name=resolved_name,
            prim_path=resolved_prim,
            tags=self.tags,
            object_type=ObjectType.RIGID,
            usd_path="",
            initial_pose=initial_pose,
            touch_force_threshold=touch_force_threshold,
        )

    def _init_event_cfg(self) -> EventTermCfg | None:
        """Reset event for this object's spawn pose.

        With ``spawn_radius > 0`` the sphere is placed at a uniform random point inside
        a ball around its initial position on every per-env reset; otherwise falls back
        to Arena's default (fixed ``Pose`` / ``PoseRange`` handling).
        """
        if self.spawn_radius > 0.0 and isinstance(self.initial_pose, Pose):
            return EventTermCfg(
                func=randomize_sphere_in_ball,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(self.name),
                    "center": self.initial_pose.position_xyz,
                    "radius": self.spawn_radius,
                },
            )
        return super()._init_event_cfg()

    def _generate_rigid_cfg(self) -> RigidObjectCfg:
        cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=_TOUCH_SPHERE_SPAWN_CFG,
            **self.asset_cfg_addon,
        )
        return self._add_initial_pose_to_cfg(cfg)
