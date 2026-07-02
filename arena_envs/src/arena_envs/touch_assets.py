"""Procedural touch-target assets for arena_envs.

Registered with Arena's ``AssetRegistry`` via ``@register_asset``. Follows the
procedural-asset convention from ``isaaclab_arena.assets.object_library``
(``ProceduralCube`` / ``ProceduralTable``): subclass ``Object`` with ``usd_path=""``
and override ``_generate_rigid_cfg`` to spawn a shape config instead of a USD file.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

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
    ):
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

    def _generate_rigid_cfg(self) -> RigidObjectCfg:
        cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=_TOUCH_SPHERE_SPAWN_CFG,
            **self.asset_cfg_addon,
        )
        return self._add_initial_pose_to_cfg(cfg)
