"""The shape-sorting task: Arena's ``SortMultiObjectTask`` with success redefined as every piece inside the box."""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import torch

from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
from isaaclab_arena.tasks.sorting_task import SortMultiObjectTask
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

from shape_sorting.predicates import objects_centers_inside_aabb


class ShapeSortingTask(SortMultiObjectTask):
    """Success when every piece's centre is inside the box cavity (``objects_centers_inside_aabb``).

    ``piece_in_box_params`` are that predicate's keyword arguments; the environment also publishes them as
    ``env.cfg.piece_in_box_params`` so policies and recorders can evaluate the same criterion themselves.
    """

    def __init__(self, *args, piece_in_box_params: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.piece_in_box_params = piece_in_box_params

    def get_termination_cfg(self) -> TaskTerminationCfg:
        # Arena's object_dropped failure and the timeout stay; only success changes.
        return replace(
            super().get_termination_cfg(),
            success=[
                ProgressObjective(
                    name="pieces_in_box",
                    predicate_sequence=[partial(objects_centers_inside_aabb, **self.piece_in_box_params)],
                )
            ],
        )


def pieces_in_box(env) -> torch.Tensor:
    """The success predicate of a running shape-sorting env, one bool per env.

    For recorders that null ``terminations.success`` (Arena's term is a stateful manager term) and poll
    success themselves.
    """
    return objects_centers_inside_aabb(env, **env.unwrapped.cfg.piece_in_box_params)
