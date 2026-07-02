"""Custom relation helpers for arena_envs.

These extend Arena's placement behaviour without modifying the vendored submodule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_arena.relations.relations import On

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


def OnWithOverhang(
    parent: "ObjectBase",
    overhang_m: float,
    *,
    relation_loss_weight: float = 1.0,
    clearance_m: float = 0.01,
) -> On:
    """An ``On`` relation that lets the child footprint hang over the parent's edges.

    This is a factory (not a new relation type) that returns a standard :class:`On`
    instance configured with a *negative* ``edge_margin_m``. Both the solver loss
    (``OnLossStrategy``) and the placement validator (``_validate_on_relations``)
    already interpret a negative margin as "the footprint may extend past each X/Y
    edge by ``|edge_margin_m|``", so oversized objects (e.g. an object larger than
    its support surface) can be placed on top.

    Returning a real ``On`` is deliberate: Arena's solver looks up loss strategies by
    exact ``type(relation)`` and skips overlap checks / validates On-relations via
    ``isinstance(rel, On)``. A brand-new subclass would have no registered loss
    strategy (there is no external hook to register one), so we reuse ``On`` itself.

    Args:
        parent: The support object the child rests on.
        overhang_m: Maximum overhang past each X/Y edge, in meters (must be >= 0).
            Choose at least half the amount by which the child exceeds the parent on
            its largest axis, plus a little slack so a placement window exists.
        relation_loss_weight: Weight for the relation loss.
        clearance_m: Safety clearance above the parent's top surface, in meters.

    Returns:
        A configured :class:`On` relation with ``edge_margin_m = -overhang_m``.
    """
    assert overhang_m >= 0.0, f"overhang_m must be non-negative, got {overhang_m}"
    # Construct with a valid (non-negative) margin to satisfy On's constructor assert,
    # then relax it to a negative value to permit overhang.
    relation = On(
        parent,
        relation_loss_weight=relation_loss_weight,
        clearance_m=clearance_m,
        edge_margin_m=0.0,
    )
    relation.edge_margin_m = -overhang_m
    return relation
