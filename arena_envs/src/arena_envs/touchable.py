"""``Touchable`` affordance for arena_envs.

Extends Arena's affordance set without modifying the vendored submodule. Follows the
built-in affordance convention (e.g. ``isaaclab_arena.affordances.pressable.Pressable``):
an affordance is a mixin combined with an ``Asset`` via multiple inheritance, so
``self.name`` / ``self.prim_path`` are provided by the asset at runtime.
"""

from __future__ import annotations

import torch
import warp as wp

from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg

from isaaclab_arena.affordances.affordance_base import AffordanceBase


# ---------------------------------------------------------------------------
# Module-level MDP term functions.
#
# These MUST be module-level (not affordance methods): Isaac Lab serializes every
# termination/reward ``func`` to a ``"module:name"`` string during the RL/Hydra
# config round-trip and reloads it with ``getattr(module, name)``. A bound method
# like ``Touchable.is_touched`` serializes to ``arena_envs.touchable:is_touched``,
# which drops the class and fails to reload. Passing these free functions (with the
# sensor name + threshold as params) mirrors how the built-in press-button task uses
# ``press_button_rl_success``.
# ---------------------------------------------------------------------------


def compute_is_touched(env: ManagerBasedEnv, sensor_name: str, force_threshold: float) -> torch.Tensor:
    """Per-env boolean tensor: is the body watched by ``sensor_name`` being touched?

    True when the total contact force on the sensor body exceeds ``force_threshold``.
    """
    sensor = env.scene.sensors[sensor_name]
    # Physics/sensor buffers are warp arrays in this stack; convert to torch like the
    # rest of the Arena codebase (e.g. object_base.get_object_pose).
    # net_forces_w: [num_envs, num_bodies, 3]
    net_forces_w = wp.to_torch(sensor.data.net_forces_w)
    force_magnitude = torch.norm(net_forces_w, dim=-1).amax(dim=1)
    return force_magnitude > force_threshold


def touched_termination(
    env: ManagerBasedEnv,
    sensor_name: str,
    force_threshold: float,
    rl_training: bool = False,
) -> torch.Tensor:
    """Success termination term. ``rl_training=True`` always returns False so success
    does not end the episode early during RL training (keeps a fixed horizon)."""
    touched = compute_is_touched(env, sensor_name, force_threshold)
    if rl_training:
        return torch.zeros_like(touched)
    return touched


def touched_reward(env: ManagerBasedEnv, sensor_name: str, force_threshold: float) -> torch.Tensor:
    """Reward term: 1.0 while touched, 0.0 otherwise."""
    return compute_is_touched(env, sensor_name, force_threshold).float()


class Touchable(AffordanceBase):
    """Affordance for objects that report when they are physically touched.

    Touch detection uses a contact sensor on the object's own rigid body:
    ``is_touched`` is True when the total contact force on the body exceeds
    ``touch_force_threshold``. Because the touch target is spawned floating and
    isolated in front of the robot, the only thing that can contact it is the
    robot's end effector, so net contact force is an effective "end effector
    touched me" signal without needing a per-body force filter.

    Affordances must be combined with an ``Asset`` via multiple inheritance (see
    ``AffordanceBase``); ``self.name`` / ``self.prim_path`` come from the asset.

    Note: MDP term configs must reference the module-level ``touched_termination`` /
    ``touched_reward`` functions (with ``sensor_name`` + ``force_threshold`` params),
    not the ``is_touched`` method below, so they survive Isaac Lab's config
    serialization. ``is_touched`` is a convenience for direct/programmatic use.
    """

    def __init__(self, touch_force_threshold: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.touch_force_threshold = touch_force_threshold

    def get_touch_sensor_name(self) -> str:
        """Scene field / sensor name for this object's touch contact sensor."""
        return f"{self.name}_touch_sensor"

    def get_touch_sensor_cfg(self) -> ContactSensorCfg:
        """Contact sensor on this object's body, updated every physics step."""
        return ContactSensorCfg(
            prim_path=self.prim_path,
            update_period=0.0,
            history_length=0,
        )

    def is_touched(self, env: ManagerBasedEnv, force_threshold: float | None = None) -> torch.Tensor:
        """Convenience wrapper around :func:`compute_is_touched` for this object."""
        threshold = force_threshold if force_threshold is not None else self.touch_force_threshold
        return compute_is_touched(env, self.get_touch_sensor_name(), threshold)

