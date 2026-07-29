"""Optional Viser view of cuRobo robot spheres + extracted collision world."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from curobo.scene import Scene
    from curobo.types import JointState


def open_collision_world_viser(
    *,
    robot_yml: str | Path,
    scene: Scene,
    joint_state: JointState | None = None,
    goal_position_xyz: list[float] | tuple[float, float, float] | None = None,
    goal_quaternion_wxyz: list[float] | tuple[float, float, float, float] | None = None,
    port: int = 8080,
):
    """Open a Viser page with obstacles, robot spheres, and optional goal frame.

    Keep the returned handle alive (store on the policy) so the server stays up.
    Open ``http://localhost:<port>`` in a browser.
    """
    from curobo.types import ContentPath, Pose
    from curobo.viewer import ViserVisualizer

    yml = Path(robot_yml).expanduser().resolve()
    viz = ViserVisualizer(
        content_path=ContentPath(robot_config_absolute_path=str(yml)),
        connect_ip="0.0.0.0",
        connect_port=port,
        add_robot_to_scene=True,
        add_control_frames=False,
        visualize_robot_spheres=True,
    )
    viz.add_scene(scene, add_control_frames=False)

    if joint_state is not None:
        viz.set_joint_state(joint_state)

    if goal_position_xyz is not None and goal_quaternion_wxyz is not None:
        goal_pose = Pose.from_list(
            list(goal_position_xyz) + list(goal_quaternion_wxyz),
        )
        viz.add_frame("/goal", goal_pose, scale=0.12)

    print(f"[curobo_viz] Viser collision debug at http://localhost:{port}")
    return viz
