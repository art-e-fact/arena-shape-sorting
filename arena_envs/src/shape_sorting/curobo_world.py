"""Build a cuRobo collision world from the live Isaac Lab USD stage.

SkillGen-style flow (adapted to cuRobo v0.8):
  1. Parse obstacle prims from the env USD subtree via ``UsdSceneParser``.
  2. Express poses in the robot base frame (``reference_prim_path``).
  3. Push the resulting ``Scene`` into ``MotionPlanner.update_world``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gymnasium as gym
    from curobo.motion_planner import MotionPlanner
    from curobo.scene import Scene

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


def env_prim_path(env_id: int = 0) -> str:
    return f"/World/envs/env_{env_id}"


def robot_prim_path(env_id: int = 0) -> str:
    return f"{env_prim_path(env_id)}/Robot"


def extract_scene_from_stage(
    stage,
    *,
    env_id: int = 0,
    ignore_substrings: list[str] | None = None,
) -> Scene:
    """Read collision geometry from ``stage`` into a cuRobo ``Scene``.

    Obstacles are transformed into the robot base frame so they match
    ``CuroboPolicy`` goals (also expressed in that frame).
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
    """Extract the USD collision world once and load it into ``planner``."""
    stage = env.unwrapped.scene.stage
    scene = extract_scene_from_stage(
        stage,
        env_id=env_id,
        ignore_substrings=ignore_substrings,
    )
    planner.update_world(scene)
    return scene
