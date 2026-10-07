"""Shape-sorting termination predicates."""

from __future__ import annotations

import torch

import warp as wp
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms


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
    unwrapped = env.unwrapped
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


STAGE_NAMES = ("untouched", "lifted", "over_box", "over_hole", "in_box")
"""Ordered progress stages for a single piece; a stage is an index into this tuple.

Binary task success (every piece in the box) sits at a few percent, which is too sparse to
compare two policies at any affordable episode count. These stages score *how far* each piece
got, turning one near-zero Bernoulli per episode into one ordinal per piece.
"""

MAX_STAGE = len(STAGE_NAMES) - 1


def read_piece_poses(env, object_names: list[str], container_name: str):
    """``(local_xyz, world_z)`` for the pieces: centres in the container frame, plus world height.

    ``local_xyz`` is ``(num_envs, num_pieces, 3)`` under the container's live root pose — the same
    transform :func:`objects_centers_inside_aabb` applies — and ``world_z`` is ``(num_envs, num_pieces)``.
    """
    unwrapped = env.unwrapped
    container: RigidObject = unwrapped.scene[container_name]
    container_pos_w = wp.to_torch(container.data.root_pos_w)
    container_quat_w = wp.to_torch(container.data.root_quat_w)

    local, world_z = [], []
    for name in object_names:
        obj_pos_w = wp.to_torch(unwrapped.scene[name].data.root_pos_w)[:, :3]
        obj_local, _ = subtract_frame_transforms(container_pos_w, container_quat_w, obj_pos_w)
        local.append(obj_local)
        world_z.append(obj_pos_w[:, 2])
    return torch.stack(local, dim=1), torch.stack(world_z, dim=1)


def piece_stages(
    local_xyz: torch.Tensor,
    world_z: torch.Tensor,
    rest_z: torch.Tensor,
    *,
    aabb_min,
    aabb_max,
    hole_centers_xy,
    hole_tolerances,
    box_xy_min,
    box_xy_max,
    lid_top_z: float,
    lift_height: float = 0.02,
    lid_margin: float = 0.01,
) -> torch.Tensor:
    """How far each piece has got right now, as ``(num_envs, num_pieces)`` indices into :data:`STAGE_NAMES`.

    Scored as the *highest stage currently satisfied* rather than as a chain that must be walked
    in order. That matters: a piece can reach the box without a recognised lift (``drop_on_hole``
    starts one already falling through its hole), and an ordered chain would score that 0 forever.

    ``rest_z`` is the lowest height seen for each piece this episode, which is its resting height —
    pieces only go up from the table, and the box cavity floor sits above it. The caller owns that
    latch and the running maximum over the episode; this function is pure.

    Args:
        local_xyz: Piece centres in the container frame, ``(num_envs, num_pieces, 3)``.
        world_z: Piece centre heights in world, ``(num_envs, num_pieces)``.
        rest_z: Resting height per piece, same shape as ``world_z``.
        aabb_min: Cavity minimum corner in the container frame (the success criterion's box).
        aabb_max: Cavity maximum corner in the container frame.
        hole_centers_xy: ``(num_pieces, 2)`` lid-hole centres, container frame, in piece order.
        hole_tolerances: ``(num_pieces,)`` max lateral distance from a piece's own hole centre.
        box_xy_min: Container footprint minimum corner in XY.
        box_xy_max: Container footprint maximum corner in XY.
        lid_top_z: Height of the lid's top surface in the container frame.
        lift_height: Metres above resting height that counts as lifted.
        lid_margin: Slack below the lid plane still counted as "on the box", for a piece
            settled into a hole rim rather than sitting flat on the lid.
    """
    device = local_xyz.device
    as_t = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)  # noqa: E731

    xy, z_local = local_xyz[..., :2], local_xyz[..., 2]
    holes, tolerances = as_t(hole_centers_xy), as_t(hole_tolerances)

    on_lid_plane = z_local >= (lid_top_z - lid_margin)
    over_box = on_lid_plane & ((xy >= as_t(box_xy_min)) & (xy <= as_t(box_xy_max))).all(dim=-1)
    over_hole = over_box & (torch.linalg.vector_norm(xy - holes, dim=-1) <= tolerances)
    inside = ((local_xyz >= as_t(aabb_min)) & (local_xyz <= as_t(aabb_max))).all(dim=-1)
    lifted = world_z > (rest_z + lift_height)

    # Applied in ascending order so the highest satisfied stage is the one that sticks.
    stage = torch.zeros_like(z_local, dtype=torch.int8)
    for reached, value in ((lifted, 1), (over_box, 2), (over_hole, 3), (inside, MAX_STAGE)):
        stage = torch.where(reached, torch.full_like(stage, value), stage)
    return stage
