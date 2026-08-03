"""Build a cuRobo collision world from the live Isaac Lab USD stage.

SkillGen-style flow (adapted to cuRobo v0.8):
  1. Parse obstacle prims from the env USD subtree via ``UsdSceneParser`` (geometry).
  2. Express poses in the robot base frame (``reference_prim_path``).
  3. Push the resulting ``Scene`` into ``MotionPlanner.update_world``.
  4. Before each plan, update *dynamic* obstacle poses from Isaac Lab rigid-body
     tensors (USD / Fabric often keeps authored xforms). Static assets (table,
     etc.) keep their USD mesh poses so child-mesh local offsets are preserved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gymnasium as gym
    import torch
    from curobo.motion_planner import MotionPlanner
    from curobo.scene import Scene
    from curobo.types import Pose

# Preallocate collision buffers so ``update_world`` can load meshes later.
DEFAULT_COLLISION_CACHE: dict[str, int] = {"mesh": 128, "cuboid": 32, "sphere": 16}

# Prim path fragments to skip (robot body, ground, cameras, debug viz).
_DEFAULT_IGNORE_SUBSTRINGS: tuple[str, ...] = (
    "/Robot",
    "/defaultGroundPlane",
    "/GroundPlane",
    "/Visuals",
    "/curobo",
    "Camera",
    "camera",
)

# Only these entities are pose-synced from Isaac Lab. Everything else (table,
# background meshes, …) keeps the USD-extracted mesh pose, which includes
# non-identity child Xforms relative to the rigid root.
#
# Procedural shape pieces / sorting box author mesh verts in the rigid-root
# frame with identity mesh Xforms, so the Lab root pose is the mesh pose.
_DYNAMIC_ENTITY_EXACT: frozenset[str] = frozenset({"sorting_box"})
_DYNAMIC_ENTITY_PREFIXES: tuple[str, ...] = ("shape_piece_",)


def env_prim_path(env_id: int = 0) -> str:
    return f"/World/envs/env_{env_id}"


def robot_prim_path(env_id: int = 0) -> str:
    return f"{env_prim_path(env_id)}/Robot"


def is_dynamic_entity(entity_name: str) -> bool:
    """Whether this Isaac Lab entity should get live pose updates."""
    if entity_name in _DYNAMIC_ENTITY_EXACT:
        return True
    return any(entity_name.startswith(p) for p in _DYNAMIC_ENTITY_PREFIXES)


def extract_scene_from_stage(
    stage,
    *,
    env_id: int = 0,
    ignore_substrings: list[str] | None = None,
) -> Scene:
    """Read collision geometry from ``stage`` into a cuRobo ``Scene``.

    Obstacles are transformed into the robot base frame so they match
    ``CuroboPolicy`` goals (also expressed in that frame).

    Prefer :func:`sync_obstacle_poses_from_env` afterward for live poses of
    dynamic objects — this USD pass is mainly for mesh / primitive geometry.
    """
    from curobo._src.util.usd_scene_parser import UsdSceneParser

    env_path = env_prim_path(env_id)
    ignore = list(ignore_substrings) if ignore_substrings is not None else list(_DEFAULT_IGNORE_SUBSTRINGS)

    parser = UsdSceneParser()
    parser.load_stage(stage)
    scene = parser.get_obstacles_from_stage(
        only_paths=[env_path],
        reference_prim_path=robot_prim_path(env_id),
        ignore_substring=ignore,
    )
    # Drop unsupported / empty types; keep mesh + primitive OBBs for checking.
    return scene.get_collision_check_world()


def obstacle_counts(scene: Scene) -> dict[str, int]:
    """Small summary for logging: ``{"mesh": N, "cuboid": M, ...}``."""
    counts: dict[str, int] = {}
    for kind in ("mesh", "cuboid", "sphere", "capsule", "cylinder"):
        items = getattr(scene, kind, None) or []
        if items:
            counts[kind] = len(items)
    return counts


def load_world_from_env(
    planner: MotionPlanner,
    env: gym.Env,
    *,
    env_id: int = 0,
    ignore_substrings: list[str] | None = None,
) -> Scene:
    """Extract USD collision geometry and load it into ``planner``.

    Call :func:`sync_obstacle_poses_from_env` after this (and before each plan)
    so dynamic object poses match Isaac Lab state.
    """
    stage = env.unwrapped.scene.stage
    scene = extract_scene_from_stage(
        stage,
        env_id=env_id,
        ignore_substrings=ignore_substrings,
    )
    planner.update_world(scene)
    return scene


def obstacle_names_for_entity(obstacle_names: list[str], entity_name: str) -> list[str]:
    """cuRobo obstacle names whose path contains ``/{entity_name}``."""
    needle = f"/{entity_name}"
    return [n for n in obstacle_names if needle in n or n.endswith(entity_name)]


def _rigid_entity_names(env: gym.Env) -> list[str]:
    """Isaac Lab rigid-object entity names (excludes the robot articulation)."""
    scene = env.unwrapped.scene
    rigid = getattr(scene, "rigid_objects", None)
    if rigid is not None:
        return list(rigid.keys())
    # Fallback: anything with root_pose_w except robot.
    names: list[str] = []
    keys = scene.keys() if hasattr(scene, "keys") else []
    for name in keys:
        if name == "robot":
            continue
        try:
            obj = scene[name]
        except Exception:
            continue
        data = getattr(obj, "data", None)
        if data is not None and hasattr(data, "root_pose_w"):
            names.append(name)
    return names


def entity_pose_in_robot_base(
    env: gym.Env,
    entity_name: str,
    *,
    device: torch.device,
) -> Pose:
    """Pose of ``entity_name`` in the robot base / URDF frame."""
    import torch
    import warp as wp
    from curobo.types import Pose
    from isaaclab.utils.math import convert_quat, subtract_frame_transforms

    robot = env.unwrapped.scene["robot"]
    try:
        obj = env.unwrapped.scene[entity_name]
    except KeyError as exc:
        available = sorted(env.unwrapped.scene.keys()) if hasattr(env.unwrapped.scene, "keys") else []
        raise KeyError(f"Entity '{entity_name}' not in scene. Available: {available}") from exc

    robot_pose_w = wp.to_torch(robot.data.root_pose_w)[0].to(device=device, dtype=torch.float32)
    obj_pose_w = wp.to_torch(obj.data.root_pose_w)[0].to(device=device, dtype=torch.float32)

    pos_b, quat_xyzw = subtract_frame_transforms(
        robot_pose_w[0:3].unsqueeze(0),
        robot_pose_w[3:7].unsqueeze(0),
        obj_pose_w[0:3].unsqueeze(0),
        obj_pose_w[3:7].unsqueeze(0),
    )
    quat_wxyz = convert_quat(quat_xyzw[0], to="wxyz")
    return Pose(
        position=pos_b[0].view(1, 3).contiguous().clone(),
        quaternion=quat_wxyz.view(1, 4).contiguous().clone(),
    )


def entity_position_in_robot_base(
    env: gym.Env,
    entity_name: str,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Entity origin in the robot base frame. Shape ``(3,)``."""
    return entity_pose_in_robot_base(env, entity_name, device=device).position.view(3)


def sync_obstacle_poses_from_env(
    planner: MotionPlanner,
    env: gym.Env,
    *,
    env_id: int = 0,
) -> Scene:
    """Update dynamic cuRobo obstacle poses from live Isaac Lab state.

    Static meshes (table, …) are left at their USD-extracted poses — forcing
    those to the rigid-root pose drops child-mesh Xforms and makes the table
    look half-buried in its own top. Dynamic procedural assets use the Lab
    root pose directly (mesh verts live in that frame).

    Returns the planner's ``scene_model`` (poses updated) for Viser refresh.
    """
    device = env.unwrapped.device
    checker = planner.scene_collision_checker
    scene_model = checker.scene_model
    if isinstance(scene_model, list):
        scene_model = scene_model[env_id]

    obstacle_names = checker.get_obstacle_names(env_id)
    sample_logs: list[str] = []
    n_updated = 0

    for entity_name in _rigid_entity_names(env):
        if not is_dynamic_entity(entity_name):
            continue
        matching = obstacle_names_for_entity(obstacle_names, entity_name)
        if not matching:
            continue

        root_pose_b = entity_pose_in_robot_base(env, entity_name, device=device)
        pose_list = root_pose_b.tolist()
        for obs_name in matching:
            checker.update_obstacle_pose(obs_name, root_pose_b, env_idx=env_id)
            obs = scene_model.get_obstacle(obs_name)
            if obs is not None:
                obs.pose = pose_list
            n_updated += 1

        pos = root_pose_b.position.view(-1)
        sample_logs.append(
            f"{entity_name}=({float(pos[0]):.3f},{float(pos[1]):.3f},{float(pos[2]):.3f})"
            f"×{len(matching)}"
        )

    if sample_logs:
        print(f"[curobo_world] synced poses ({n_updated} obstacles): {', '.join(sample_logs)}")
    elif n_updated == 0:
        dynamic = [n for n in _rigid_entity_names(env) if is_dynamic_entity(n)]
        print(
            "[curobo_world] WARNING: no dynamic obstacles synced; "
            f"dynamic_entities={dynamic} "
            f"obstacles_sample={obstacle_names[:8]}"
        )
    return scene_model
