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
| `so101_abs_joint` | Absolute joint positions (leader + joint-space gamepad) |
| `so101_rel_joint` | Relative joint positions |
| `so101_ik` | Relative SE(3) differential IK + binary Jaw (keyboard / gamepad / spacemouse) |

USD joints: `Rotation`, `Pitch`, `Elbow`, `Wrist_Pitch`, `Wrist_Roll`, `Jaw`.
The robot USD comes from the [Sim-to-Real-SO-101-Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop).

Wrist camera is a Python `CameraCfg` on `Robot/gripper/gripper_cam`
(enabled with `enable_cameras=True`). Shape-sorting also enables a fixed
env-frame `external_camera` (over-shoulder / table view) beside the wrist cam.
Tune the external view by overriding `camera_config.external_camera.offset`
(`pos` + `rot`); there is no prim look-at on `CameraCfg` — shape-sorting uses
an eye/target helper to derive the quaternion.

`so101_ik` is a 5-DOF arm: DLS tracks EE position and does best-effort orientation on the 6D pose command.

## cuRobo planning assets

Generate a URDF (from the workshop USD) plus a cuRobo robot YAML (collision spheres,
self-collision ignore matrix, locked Jaw, home pose) under
`embodiments/data/curobo/`:

```bash
# Inside the Isaac Sim / Arena env (needs CUDA + nvidia-curobo)
python -m arena_so101.generate_curobo_config --headless
```

Outputs:

| File | Purpose |
|------|---------|
| `embodiments/data/curobo/urdf/SO-ARM101-USD.urdf` | Kinematics matching sim joint names |
| `embodiments/data/curobo/meshes/` | Link meshes referenced by the URDF |
| `embodiments/data/curobo/so101.yml` | cuRobo `robot_cfg` for `MotionPlanner` |

Rebuild spheres from an existing URDF (no Isaac Sim):

```bash
python -m arena_so101.generate_curobo_config \
  --skip-usd-convert \
  --urdf arena_so101/src/arena_so101/embodiments/data/curobo/urdf/SO-ARM101-USD.urdf \
  --asset-path arena_so101/src/arena_so101/embodiments/data/curobo/meshes
```

Add `--visualize` to inspect fitted spheres in Viser.

## LeRobot recording

`shape_sorting.generate_policy_demos` records successful scripted-policy
rollouts directly as LeRobot v3. It stores six-joint state/action vectors and
both the wrist and exterior RGB cameras. Failed attempts are discarded before
they enter the dataset.

```bash
python -m shape_sorting.generate_policy_demos \
  --policy_type shape_sorting.curobo_policy.CuroboPolicy \
  --generation_num_trials 10 \
  --output_dir ./datasets/curobo_shape_sorting \
  --dataset_repo_id local/curobo_shape_sorting \
  shape_sorting_test --embodiment so101_abs_joint
```

The output directory must not exist by default. Use `--resume` to append to a
compatible dataset or `--overwrite` to recreate it. Camera rendering and
streaming video encoding are enabled automatically; use
`--disable_streaming_encoding` to encode from temporary PNGs instead.

The YAML/JSON assets and `convert_hdf5_to_lerobot` module under
`arena_so101.lerobot` are legacy support for existing HDF5/GR00T datasets.
Direct LeRobot v3 recording does not use them.

## Teleop

```bash
# Joint-space gamepad (absolute joints — recommended for SO-101 demos)
python -m shape_sorting.run_teleop \
  --viz kit --num_envs 1 \
  shape_sorting_test \
  --embodiment so101_abs_joint \
  --teleop_device gamepad

# SE(3) task-space (gamepad / keyboard / spacemouse)
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

### Joint-space gamepad layout (`so101_abs_joint` + `gamepad`)

Sticks/triggers integrate into a held absolute joint target. Releasing sticks holds pose.

| Input | Joint |
|-------|-------|
| RT (+) / LT (−) | `Rotation` (base yaw) |
| Left stick up/down | `Pitch` |
| Left stick left/right | `Elbow` |
| Right stick up/down | `Wrist_Pitch` |
| Right stick right/left | `Wrist_Roll` |
| X | `Jaw` toggle open / close (absolute limits) |

Speed is `delta_scale` on `GamepadCfg` (default `0.03` rad/step at full deflection).

`so101_leader` emits a (6,) absolute joint vector for `so101_abs_joint`. Also works with
Arena's `record_demos.py` when the env wires `--teleop_device so101_leader`.
