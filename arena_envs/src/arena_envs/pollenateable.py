"""``Pollenateable`` affordance for arena_envs.

Discovers ``approach_frame_*`` Xform prims in a plant USD and exposes them as
logical approach sites. Per-episode "reached" latch state lives in the task MDP
terms (see ``pollenate_task``), not on the affordance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from pxr import Usd, UsdGeom

from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.sim.utils import get_current_stage

from isaaclab_arena.affordances.affordance_base import AffordanceBase
from isaaclab_arena.utils.usd_helpers import open_stage

_APPROACH_FRAME_RE = re.compile(r"^approach_frame_(\d+)$")


@dataclass(frozen=True)
class ApproachFrame:
    """Metadata for one approach site on a pollinateable plant."""

    name: str
    suffix: str
    index: int


def _env_prim_path(env: ManagerBasedEnv, plant_name: str, frame_suffix: str, env_id: int) -> str:
    return f"{env.scene.env_ns}/env_{env_id}/{plant_name}{frame_suffix}"


def _get_prim_world_pose(stage: Usd.Stage, prim_path: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world pose as position (3,) and quaternion (w, x, y, z)."""
    prim = stage.GetPrimAtPath(prim_path)
    assert prim.IsValid(), f"No prim at path {prim_path}"
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    prim_tf = xform_cache.GetLocalToWorldTransform(prim)
    prim_tf.Orthonormalize()
    translation = prim_tf.ExtractTranslation()
    orientation = prim_tf.ExtractRotationQuat()
    pos = torch.tensor(translation, dtype=torch.float32, device=device)
    quat = torch.tensor(
        [orientation.GetReal(), *orientation.GetImaginary()],
        dtype=torch.float32,
        device=device,
    )
    return pos, quat


def get_frame_pos_w(env: ManagerBasedEnv, plant_name: str, frame_suffix: str) -> torch.Tensor:
    """World-frame position (num_envs, 3) of an approach frame under a spawned plant."""
    stage = get_current_stage()
    positions = [
        _get_prim_world_pose(stage, _env_prim_path(env, plant_name, frame_suffix, env_id), env.device)[0]
        for env_id in range(env.num_envs)
    ]
    return torch.stack(positions, dim=0)


def get_frame_world_poses(
    env: ManagerBasedEnv, plant_name: str, frame_suffix: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """World-frame pose as position (num_envs, 3) and quaternion (num_envs, 4) in (w, x, y, z)."""
    stage = get_current_stage()
    positions = []
    orientations = []
    for env_id in range(env.num_envs):
        pos, quat = _get_prim_world_pose(
            stage, _env_prim_path(env, plant_name, frame_suffix, env_id), env.device
        )
        positions.append(pos)
        orientations.append(quat)
    return torch.stack(positions, dim=0), torch.stack(orientations, dim=0)


def get_frame_pose_w(env: ManagerBasedEnv, plant_name: str, frame_suffix: str) -> torch.Tensor:
    """World-frame pose (num_envs, 7) as ``(x, y, z, qw, qx, qy, qz)``."""
    pos_w, quat_wxyz = get_frame_world_poses(env, plant_name, frame_suffix)
    return torch.cat([pos_w, quat_wxyz], dim=-1)


def discover_approach_frame_paths(usd_path: str) -> list[tuple[str, int]]:
    """Scan a plant USD for ``approach_frame_<N>`` prims.

    Returns sorted ``(suffix, index)`` pairs where *suffix* is the path relative to
    the USD default prim (e.g. ``/Petiole_003/approach_frame_0``).
    """
    with open_stage(usd_path) as stage:
        default_prim = stage.GetDefaultPrim()
        if not default_prim:
            raise ValueError(f"No default prim in {usd_path}")
        default_path = str(default_prim.GetPath())
        matches: list[tuple[str, int]] = []
        for prim in stage.Traverse():
            m = _APPROACH_FRAME_RE.match(prim.GetName())
            if m is None:
                continue
            full_path = str(prim.GetPath())
            suffix = full_path.removeprefix(default_path)
            matches.append((suffix, int(m.group(1))))
        matches.sort(key=lambda item: item[1])
        return matches


class Pollenateable(AffordanceBase):
    """Affordance for plants with named ``approach_frame_*`` pollinate sites.

    Combined with a ``LibraryObject`` (or similar) via multiple inheritance.
    ``get_approach_frames()`` lazily discovers one ``ApproachFrame`` per prim in
    the plant USD. Poses are read from the spawned plant at runtime via USD.
    """

    def __init__(self, reach_distance_threshold: float = 0.005, **kwargs):
        super().__init__(**kwargs)
        self.reach_distance_threshold = reach_distance_threshold
        self._approach_frames: list[ApproachFrame] | None = None

    def get_approach_frames(self) -> list[ApproachFrame]:
        """Return metadata for every ``approach_frame_*`` on this plant."""
        if self._approach_frames is None:
            self._approach_frames = self._build_approach_frames()
        return self._approach_frames

    def _build_approach_frames(self) -> list[ApproachFrame]:
        usd_path = getattr(self, "usd_path", None)
        assert usd_path, f"{self.__class__.__name__} must define usd_path for frame discovery"
        discovered = discover_approach_frame_paths(usd_path)
        assert len(discovered) > 0, f"No approach_frame_* prims found in {usd_path}"
        frames = [
            ApproachFrame(name=f"{self.name}_approach_{index}", suffix=suffix, index=index)
            for suffix, index in discovered
        ]
        return frames
