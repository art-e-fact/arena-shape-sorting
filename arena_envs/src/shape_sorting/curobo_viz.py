"""Debug visualization sinks for cuRobo motion planning.

``CollisionDebugViz`` is the backend-agnostic interface used by ``MotionClient``.
Swap ``ViserCollisionDebugViz`` for a Rerun (or other) implementation later
without touching planning / policy code.

Kit viewport frame markers are a separate Isaac-only sink (``KitFrameMarkers``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import gymnasium as gym
    from curobo.scene import Scene
    from curobo.types import JointState


# ---------------------------------------------------------------------------
# Backend-agnostic collision / robot debug sink
# ---------------------------------------------------------------------------


@runtime_checkable
class CollisionDebugViz(Protocol):
    """Optional side-channel for collision world + robot state.

    Implementations must be safe no-ops when not yet opened / unavailable.
    """

    def set_world(self, scene: Scene) -> None:
        """Replace / refresh collision obstacle meshes."""

    def set_robot_state(self, joint_state: JointState) -> None:
        """Push active planner joint state (and refresh attached spheres if any)."""

    def set_goal(
        self,
        position_xyz: list[float] | tuple[float, float, float],
        quaternion_wxyz: list[float] | tuple[float, float, float, float],
    ) -> None:
        """Show the current EE goal pose (robot-base / world frame)."""

    def set_attached_spheres(self, spheres_xyzr: Any) -> None:
        """Draw attached-object spheres as ``(N, 4)`` ``(x, y, z, r)`` in world."""

    def clear_attached_spheres(self) -> None:
        """Remove attached-object sphere visuals."""

    def close(self) -> None:
        """Release backend resources."""


class NullCollisionDebugViz:
    """No-op sink used when collision debug is disabled."""

    def set_world(self, scene: Scene) -> None:
        return

    def set_robot_state(self, joint_state: JointState) -> None:
        return

    def set_goal(self, position_xyz, quaternion_wxyz) -> None:
        return

    def set_attached_spheres(self, spheres_xyzr: Any) -> None:
        return

    def clear_attached_spheres(self) -> None:
        return

    def close(self) -> None:
        return


class CompositeCollisionDebugViz:
    """Fan-out to several sinks (e.g. Viser today + Rerun later)."""

    def __init__(self, *sinks: CollisionDebugViz):
        self._sinks = list(sinks)

    def set_world(self, scene: Scene) -> None:
        for s in self._sinks:
            s.set_world(scene)

    def set_robot_state(self, joint_state: JointState) -> None:
        for s in self._sinks:
            s.set_robot_state(joint_state)

    def set_goal(self, position_xyz, quaternion_wxyz) -> None:
        for s in self._sinks:
            s.set_goal(position_xyz, quaternion_wxyz)

    def set_attached_spheres(self, spheres_xyzr: Any) -> None:
        for s in self._sinks:
            s.set_attached_spheres(spheres_xyzr)

    def clear_attached_spheres(self) -> None:
        for s in self._sinks:
            s.clear_attached_spheres()

    def close(self) -> None:
        for s in self._sinks:
            s.close()


# ---------------------------------------------------------------------------
# Viser backend
# ---------------------------------------------------------------------------

# Same naming scheme as ``ViserVisualizer.add_scene`` (poses baked into vertices).
_OBSTACLE_MESH_ATTR = "_shape_sorting_obstacle_mesh_names"
_ATTACHED_SPHERES_NAME = "/attached_object_spheres"
_ATTACHED_SPHERES_COLOR = (220, 90, 40)  # Distinct from robot collision spheres.


def _obstacle_mesh_name(mesh_name: str) -> str:
    return "/obstacles/" + mesh_name + "/mesh"


def _as_numpy_xyzr(spheres_xyzr: Any) -> np.ndarray:
    if hasattr(spheres_xyzr, "detach"):
        return spheres_xyzr.detach().float().cpu().numpy()
    return np.asarray(spheres_xyzr, dtype=np.float64)


def _update_obstacle_meshes(viz: Any, scene: Scene) -> None:
    """Replace Viser obstacle meshes; poses are baked into vertices."""
    from curobo._src.geom.types import SceneCfg

    scene_api = viz._server.scene
    for name in getattr(viz, _OBSTACLE_MESH_ATTR, ()):
        if hasattr(scene_api, "remove_by_name"):
            scene_api.remove_by_name(name)

    mesh_scene = SceneCfg.create_mesh_scene(scene)
    names: list[str] = []
    for mesh in mesh_scene.mesh or []:
        name = _obstacle_mesh_name(mesh.name)
        viz.add_mesh(mesh.get_trimesh_mesh(transform_with_pose=True), name=name)
        names.append(name)
    setattr(viz, _OBSTACLE_MESH_ATTR, names)


def _set_attached_spheres(viz: Any, spheres_xyzr: Any) -> None:
    arr = _as_numpy_xyzr(spheres_xyzr)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 4:
        _clear_attached_spheres(viz)
        return

    # Drop padding slots (cuRobo uses large negative radii for unused spheres).
    arr = arr[arr[:, 3] > 1e-4]
    if arr.shape[0] == 0:
        _clear_attached_spheres(viz)
        return

    viz.add_batched_spheres_from_position(
        position=arr[:, :3].copy(),
        radius=arr[:, 3].copy(),
        color=_ATTACHED_SPHERES_COLOR,
        name=_ATTACHED_SPHERES_NAME,
    )


def _clear_attached_spheres(viz: Any) -> None:
    scene_api = viz._server.scene
    if hasattr(scene_api, "remove_by_name"):
        scene_api.remove_by_name(_ATTACHED_SPHERES_NAME)


class ViserCollisionDebugViz:
    """Viser page: collision meshes, robot spheres, goal frame, attached spheres.

    Opens lazily on the first ``set_robot_state`` once a world scene is known
    (matches the previous policy behavior). Keep the instance alive so the
    HTTP server stays up.
    """

    def __init__(self, robot_yml: str | Path, *, port: int = 8080):
        self._robot_yml = Path(robot_yml).expanduser().resolve()
        self._port = int(port)
        self._viz: Any | None = None
        self._pending_scene: Scene | None = None
        self._pending_goal: tuple[list[float], list[float]] | None = None

    def set_world(self, scene: Scene) -> None:
        if self._viz is None:
            self._pending_scene = scene
            return
        _update_obstacle_meshes(self._viz, scene)

    def set_robot_state(self, joint_state: JointState) -> None:
        self._ensure_open(joint_state)
        if self._viz is not None:
            self._viz.set_joint_state(joint_state)

    def set_goal(self, position_xyz, quaternion_wxyz) -> None:
        pos = list(position_xyz)
        quat = list(quaternion_wxyz)
        if self._viz is None:
            self._pending_goal = (pos, quat)
            return
        from curobo.types import Pose

        self._viz.add_frame("/goal", Pose.from_list(pos + quat), scale=0.12)

    def set_attached_spheres(self, spheres_xyzr: Any) -> None:
        if self._viz is not None:
            _set_attached_spheres(self._viz, spheres_xyzr)

    def clear_attached_spheres(self) -> None:
        if self._viz is not None:
            _clear_attached_spheres(self._viz)

    def close(self) -> None:
        self._viz = None
        self._pending_scene = None
        self._pending_goal = None

    def _ensure_open(self, joint_state: JointState) -> None:
        if self._viz is not None or self._pending_scene is None:
            return
        from curobo.types import ContentPath, Pose
        from curobo.viewer import ViserVisualizer

        self._viz = ViserVisualizer(
            content_path=ContentPath(robot_config_absolute_path=str(self._robot_yml)),
            connect_ip="0.0.0.0",
            connect_port=self._port,
            add_robot_to_scene=True,
            add_control_frames=False,
            visualize_robot_spheres=True,
        )
        _update_obstacle_meshes(self._viz, self._pending_scene)
        self._viz.set_joint_state(joint_state)

        if self._pending_goal is not None:
            pos, quat = self._pending_goal
            self._viz.add_frame("/goal", Pose.from_list(pos + quat), scale=0.12)
            self._pending_goal = None

        self._pending_scene = None
        print(f"[curobo_viz] Viser collision debug at http://localhost:{self._port}")


# ---------------------------------------------------------------------------
# Isaac Kit viewport frame markers (goal + EE)
# ---------------------------------------------------------------------------


class KitFrameMarkers:
    """Draw goal + EE frame axes in the Kit viewport."""

    def __init__(self, *, scale: float = 0.08):
        self._scale = float(scale)
        self._goal_marker = None
        self._ee_marker = None
        self._ee_body_id: int | None = None

    def ensure(self, env: gym.Env, tool_frame: str) -> None:
        if self._goal_marker is not None and self._ee_marker is not None:
            return

        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG

        s = self._scale
        goal_cfg = FRAME_MARKER_CFG.copy()
        goal_cfg.prim_path = "/Visuals/CuroboPolicy/goal_frame"
        goal_cfg.markers["frame"].scale = (s, s, s)
        self._goal_marker = VisualizationMarkers(goal_cfg)

        ee_cfg = FRAME_MARKER_CFG.copy()
        ee_cfg.prim_path = "/Visuals/CuroboPolicy/ee_frame"
        ee_cfg.markers["frame"].scale = (s, s, s)
        self._ee_marker = VisualizationMarkers(ee_cfg)

        robot = env.unwrapped.scene["robot"]
        body_ids, body_names = robot.find_bodies(tool_frame)
        if not body_ids:
            raise RuntimeError(
                f"KitFrameMarkers: robot has no body matching tool frame '{tool_frame}'. "
                f"Have: {list(robot.body_names)}"
            )
        self._ee_body_id = int(body_ids[0])
        print(f"[curobo_viz] Kit markers ready (goal + EE '{body_names[0]}').")

    def update(
        self,
        env: gym.Env,
        device: Any,
        *,
        goal_pos_w: Any,
        goal_quat_xyzw: Any,
    ) -> None:
        import torch
        import warp as wp

        assert self._goal_marker is not None and self._ee_marker is not None
        assert self._ee_body_id is not None

        self._goal_marker.visualize(translations=goal_pos_w, orientations=goal_quat_xyzw)

        robot = env.unwrapped.scene["robot"]
        ee_pose_w = wp.to_torch(robot.data.body_link_pose_w)[:, self._ee_body_id, :].to(
            device=device, dtype=torch.float32
        )
        self._ee_marker.visualize(translations=ee_pose_w[:, 0:3], orientations=ee_pose_w[:, 3:7])

    def close(self) -> None:
        self._goal_marker = None
        self._ee_marker = None
        self._ee_body_id = None
