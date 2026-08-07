"""Shape-sorting termination predicates."""

from __future__ import annotations

import torch

import warp as wp
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_arena.tasks.predicates.predicate_utils import get_env


def objects_centers_inside_aabb(
    env: ManagerBasedRLEnv,
    object_cfg_list: list[SceneEntityCfg],
    container_cfg: SceneEntityCfg,
    aabb_min: tuple[float, float, float],
    aabb_max: tuple[float, float, float],
    velocity_threshold: float = 0.1,
) -> torch.Tensor:
    """True when every object's root center lies inside a container-local AABB and is settled.

    ``aabb_min`` / ``aabb_max`` are axis-aligned extents in the container's local frame
    (e.g. the sorting-box cavity). Each object pose is transformed into that frame via
    the container's live root pose.
    """
    unwrapped = get_env(env)
    container: RigidObject = unwrapped.scene[container_cfg.name]
    container_pos_w = wp.to_torch(container.data.root_pos_w)
    container_quat_w = wp.to_torch(container.data.root_quat_w)

    min_t = torch.tensor(aabb_min, device=unwrapped.device, dtype=torch.float32)
    max_t = torch.tensor(aabb_max, device=unwrapped.device, dtype=torch.float32)

    condition = torch.ones(unwrapped.num_envs, device=unwrapped.device, dtype=torch.bool)
    for object_cfg in object_cfg_list:
        obj: RigidObject = unwrapped.scene[object_cfg.name]
        obj_pos_w = wp.to_torch(obj.data.root_pos_w)[:, :3]
        obj_pos_local, _ = subtract_frame_transforms(container_pos_w, container_quat_w, obj_pos_w)
        inside = torch.all((obj_pos_local >= min_t) & (obj_pos_local <= max_t), dim=-1)

        speed = torch.linalg.vector_norm(wp.to_torch(obj.data.root_lin_vel_w), dim=-1)
        settled = speed < velocity_threshold
        condition = torch.logical_and(condition, torch.logical_and(inside, settled))
    return condition
