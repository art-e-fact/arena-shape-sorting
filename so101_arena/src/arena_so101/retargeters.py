"""Retargeter entries pairing so101_leader with SO-101 joint embodiments."""

from __future__ import annotations

from collections.abc import Callable

from isaaclab_arena.assets.register import register_retargeter
from isaaclab_arena.assets.retargeter_library import RetargetterBase


@register_retargeter
class SO101LeaderAbsJointRetargeter(RetargetterBase):
    device = "so101_leader"
    embodiment = "so101_abs_joint"

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        return None


@register_retargeter
class SO101LeaderRelJointRetargeter(RetargetterBase):
    device = "so101_leader"
    embodiment = "so101_rel_joint"

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        return None
