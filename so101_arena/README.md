# arena-so101

Reusable SO-101 follower embodiment (and optional leader-arm helpers) for
[Isaac Lab Arena](https://github.com/isaac-sim/IsaacLab-Arena).

## Install

```bash
/isaac-sim/python.sh -m pip install -e so101_arena
# optional: leader teleop
/isaac-sim/python.sh -m pip install -e "so101_arena[leader]"
```

After ``SimulationApp`` is running, register once:

```python
import arena_so101
arena_so101.register()  # so101_abs_joint, so101_rel_joint, so101_ik, so101_leader, gamepad
```

Then use like any Arena embodiment:

```python
embodiment = asset_registry.get_asset_by_name("so101_abs_joint")(enable_cameras=True)
```

## Embodiments

| Name | Actions |
|------|---------|
| `so101_abs_joint` | Absolute joint positions (best for leader teleop) |
| `so101_rel_joint` | Relative joint positions |
| `so101_ik` | Relative SE(3) differential IK + binary Jaw (keyboard / gamepad / spacemouse) |

USD joints (workshop naming): `Rotation`, `Pitch`, `Elbow`, `Wrist_Pitch`, `Wrist_Roll`, `Jaw`.

Wrist camera is a Python `TiledCameraCfg` on `Robot/gripper/gripper_cam` (enabled with `enable_cameras=True`).

`so101_ik` is a 5-DOF arm: DLS tracks EE position and does best-effort orientation on the 6D pose command.

## Teleop

Same Arena teleop path for SE(3) devices and the physical leader:

```bash
# SE(3)
python -m shape_sorting.run_teleop \
  --viz kit --num_envs 1 \
  shape_sorting_test \
  --embodiment so101_ik \
  --teleop_device gamepad   # or keyboard / spacemouse

# Physical SO-101 leader → abs joints (needs so101_arena[leader])
python -m shape_sorting.run_teleop \
  --viz kit --num_envs 1 \
  shape_sorting_test \
  --embodiment so101_abs_joint \
  --teleop_device so101_leader \
  --leader_port /dev/ttyACM0
```

`so101_leader` is a normal Arena/Isaac Lab device: `advance()` returns a (6,) absolute
joint vector for `so101_abs_joint`. Also works with Arena's `record_demos.py` when the
env wires `--teleop_device so101_leader`.
