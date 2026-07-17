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
arena_so101.register()  # so101_abs_joint, so101_rel_joint, so101_leader
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

USD joints (workshop naming): `Rotation`, `Pitch`, `Elbow`, `Wrist_Pitch`, `Wrist_Roll`, `Jaw`.

Wrist camera is a Python `TiledCameraCfg` on `Robot/gripper/gripper_cam` (enabled with `enable_cameras=True`).

## Leader teleop

```bash
python -m arena_so101.teleop_leader --port /dev/ttyACM0 --id leader \
  --env_name <your_registered_env> ...
```

Maps LeRobot leader degrees → sim radians by joint index (same order as the workshop).
