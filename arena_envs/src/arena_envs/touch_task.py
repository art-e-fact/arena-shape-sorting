"""Touch tasks for arena_envs.

``TouchTask`` (imitation flavour) and ``TouchTaskRL`` (adds observations, rewards and
RL-mode success suppression). Modelled on ``isaaclab_arena.tasks.press_button_task``.
Success = the touch target is touched (see :class:`arena_envs.touchable.Touchable`).
"""

from __future__ import annotations

import numpy as np
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.observations import observations
from isaaclab_arena.tasks.rewards import rewards
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object
from isaaclab_arena.utils.configclass import make_configclass

from arena_envs.touchable import Touchable, touched_reward, touched_termination


@configclass
class TouchTerminationsCfg:
    """Termination terms for the Touch task."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    # Provided by the task at construction time (depends on the touch object).
    success: TerminationTermCfg = MISSING


@register_task
class TouchTask(TaskBase):
    """Touch a floating target. Success = the object is touched."""

    def __init__(
        self,
        touch_object: Touchable,
        episode_length_s: float = 5.0,
        task_description: str | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)
        assert isinstance(touch_object, Touchable), "touch_object must be an instance of Touchable"
        self.touch_object = touch_object
        self.task_description = (
            f"Touch the {touch_object.name}" if task_description is None else task_description
        )

    def get_scene_cfg(self):
        # Contribute the object's touch contact sensor to the scene.
        sensor_name = self.touch_object.get_touch_sensor_name()
        sensor_cfg = self.touch_object.get_touch_sensor_cfg()
        SceneCfg = make_configclass("TouchSceneCfg", [(sensor_name, ContactSensorCfg, sensor_cfg)])
        return SceneCfg()

    def get_termination_cfg(self):
        success = TerminationTermCfg(
            func=touched_termination,
            params={
                "sensor_name": self.touch_object.get_touch_sensor_name(),
                "force_threshold": self.touch_object.touch_force_threshold,
            },
        )
        return TouchTerminationsCfg(success=success)

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("Function not implemented yet.")

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.touch_object,
            offset=np.array([-1.5, -1.5, 1.5]),
        )


@register_task
class TouchTaskRL(TouchTask):
    """RL version of :class:`TouchTask` with reach + touch rewards."""

    def __init__(
        self,
        touch_object: Touchable,
        embodiment: EmbodimentBase,
        episode_length_s: float = 5.0,
        rl_training_mode: bool = True,
        task_description: str | None = None,
    ):
        """Initialize the Touch RL task.

        Args:
            touch_object: The touchable target.
            embodiment: The robot embodiment.
            episode_length_s: Episode length in seconds.
            rl_training_mode: If True, disables success termination during training.
            task_description: Optional override for the task description.
        """
        self.rl_training_mode = rl_training_mode
        self.embodiment = embodiment

        super().__init__(
            touch_object=touch_object,
            episode_length_s=episode_length_s,
            task_description=task_description,
        )

        robot_name = self.embodiment.get_embodiment_name_in_scene()
        self.observation_cfg = TouchObservationsCfg(touch_object=self.touch_object, robot_name=robot_name)
        self.rewards_cfg = TouchRewardCfg(
            touch_object=self.touch_object,
            ee_frame_name=self.embodiment.get_ee_frame_name(self.embodiment.get_arm_mode()),
        )
        self.termination_cfg = self.make_rl_termination_cfg()

    def make_rl_termination_cfg(self):
        """Success termination that is suppressed while ``rl_training_mode`` is on."""
        success = TerminationTermCfg(
            func=touched_termination,
            params={
                "sensor_name": self.touch_object.get_touch_sensor_name(),
                "force_threshold": self.touch_object.touch_force_threshold,
                "rl_training": self.rl_training_mode,
            },
        )
        return TouchTerminationsCfg(success=success)

    def get_observation_cfg(self):
        return self.observation_cfg

    def get_rewards_cfg(self):
        return self.rewards_cfg

    def get_termination_cfg(self):
        return self.termination_cfg


@configclass
class TouchObservationsCfg:
    """Observation specifications for the Touch task."""

    task_obs: ObsGroup = MISSING

    def __init__(self, touch_object: Touchable, robot_name: str):
        @configclass
        class TaskObsCfg(ObsGroup):
            object_position = ObsTerm(
                func=observations.object_position_in_frame,
                params={
                    "root_frame_cfg": SceneEntityCfg(robot_name),
                    "object_cfg": SceneEntityCfg(touch_object.name),
                },
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        self.task_obs = TaskObsCfg()


@configclass
class TouchRewardCfg:
    """Reward terms for the Touch task: reach the target, then touch it."""

    reaching_object: RewardTermCfg = MISSING
    touching_object: RewardTermCfg = MISSING

    def __init__(self, touch_object: Touchable, ee_frame_name: str):
        self.reaching_object = RewardTermCfg(
            func=rewards.object_ee_distance,
            params={
                "std": 0.1,
                "object_cfg": SceneEntityCfg(touch_object.name),
                "ee_frame_cfg": SceneEntityCfg(ee_frame_name),
            },
            weight=1.0,
        )
        self.touching_object = RewardTermCfg(
            func=touched_reward,
            params={
                "sensor_name": touch_object.get_touch_sensor_name(),
                "force_threshold": touch_object.touch_force_threshold,
            },
            weight=5.0,
        )
