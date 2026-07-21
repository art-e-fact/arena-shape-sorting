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

## SE(3) teleop

```bash
python -m shape_sorting.run_teleop \
  --viz kit --num_envs 1 \
  shape_sorting_test \
  --embodiment so101_ik \
  --teleop_device gamepad   # or keyboard / spacemouse
```

## Leader teleop

```bash
python -m arena_so101.teleop_leader --port /dev/ttyACM0 --id leader \
  --env_name <your_registered_env> ...
```

Maps LeRobot leader degrees → sim radians by joint index (same order as the workshop).
Use a joint embodiment (`so101_abs_joint`), not `so101_ik`.
