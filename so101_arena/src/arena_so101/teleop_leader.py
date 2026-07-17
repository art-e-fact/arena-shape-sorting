"""Leader-arm helpers: physical SO-101 leader → sim follower (abs joints).

Arena's SE3 ``teleop.py`` cannot drive this device. Build an env with
``so101_abs_joint``, then::

    from arena_so101.teleop_leader import run_leader_loop
    run_leader_loop(env, port="/dev/ttyACM0")
"""

from __future__ import annotations

import argparse

import torch


def open_leader(port: str, leader_id: str = "leader"):
    """Connect a LeRobot SO101Leader. Requires ``pip install lerobot``."""
    try:
        from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
    except ImportError as exc:
        raise ImportError(
            "Leader teleop needs lerobot. Install with: pip install -e 'so101_arena[leader]'"
        ) from exc

    cfg = SO101LeaderConfig(port=port, id=leader_id)
    leader = SO101Leader(cfg)
    leader.connect()
    return leader


def leader_to_action_tensor(leader, env_device: torch.device) -> torch.Tensor:
    """Read leader joints and map to a (6,) sim action (radians)."""
    from arena_so101.mapping import leader_dict_to_sim_radians

    return leader_dict_to_sim_radians(leader.get_action(), device=env_device)


def run_leader_loop(env, *, port: str, leader_id: str = "leader") -> None:
    """Step ``env`` with absolute joint actions from the physical leader."""
    leader = open_leader(port, leader_id)
    device = env.unwrapped.device
    actions = torch.zeros(env.action_space.shape, device=device)

    try:
        env.reset()
        while True:
            actions[:] = leader_to_action_tensor(leader, device)
            env.step(actions)
    finally:
        leader.disconnect()
        env.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--id", type=str, default="leader", dest="leader_id")
    args = parser.parse_args(argv)
    print(
        "Build an Arena env with embodiment 'so101_abs_joint', then:\n"
        "  from arena_so101.teleop_leader import run_leader_loop\n"
        f"  run_leader_loop(env, port={args.port!r}, leader_id={args.leader_id!r})\n"
    )


if __name__ == "__main__":
    main()
