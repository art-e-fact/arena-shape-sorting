# envhub — LeRobot EnvHub package for our Arena environments

Lets the stock LeRobot CLI (`lerobot-eval`) roll out a policy in this repo's Isaac Lab
Arena environments, without wrapping or forking any LeRobot script.

| File | Purpose |
| --- | --- |
| `config.py` | `--env.type=shape_sorting_arena`: a `lerobot.envs.IsaaclabArenaEnv` subclass with our defaults |
| `env.py` | `make_env()` — the EnvHub entry point: launches Isaac Sim and builds the Arena env |
| `isaaclab_env_wrapper.py` | Adapts the batched Isaac Lab env to LeRobot's vector-env API |
| `scripts/zero_action_rollout.py` | Smoke test: builds the env and steps zero actions, no policy |

LeRobot discovers `shape_sorting_arena` through draccus plugin discovery
(`--env.discover_packages_path=envhub`), so `envhub` must be importable. It is not an
installed package, so run from the repo root with `PYTHONPATH=.`.

## Evaluate a policy

```bash
PYTHONPATH=. lerobot-eval \
  --policy.path=Artefacts/act-shape-sorting-so101 \
  --policy.device=cuda \
  --env.discover_packages_path=envhub \
  --env.type=shape_sorting_arena \
  --env.visualizer=kit \
  --rename_map='{"observation.images.camera_ego_rgb": "observation.images.ego_view",
                 "observation.images.external_camera_rgb": "observation.images.exterior_image"}' \
  --eval.batch_size=1 \
  --eval.n_episodes=10
```

`--rename_map` bridges the naming gap: `env.py` exposes Arena's observation term names
(`camera_ego_rgb`), while the policy was trained on the dataset's names (`ego_view`).
Drop `--env.visualizer=kit` to run headless.

Useful overrides:

- `--eval.batch_size=N` — N sub-environments batched inside the one simulation. Isaac Lab
  vectorizes on the GPU regardless of `--eval.use_async_envs` (default `true`), so that
  flag is simply ignored — no need to set it to `false`.
- `--env.episode_length=N` — cap rollout steps (default: the Arena task's own length).
- `--env.kwargs='{"piece_size": 0.035}'` — any other field of the environment's own
  config dataclass (`ShapeSortingEnvironmentCfg`); only `embodiment`, `enable_cameras`
  and `teleop_device` have first-class `--env.*` flags.

## Training

Training never touches the environment: `lerobot-train` consumes a `LeRobotDataset`
recorded from simulation by `arena_so101.lerobot.recorder`. Only evaluation needs `envhub`.

## Adding an environment

Add an entry to `ENVIRONMENTS` in `env.py` (module, factory class, its config dataclass),
then a small `EnvConfig` subclass in `config.py` carrying the observation layout the
policy expects.

## Publishing to the Hub (optional)

`env.py` is also loadable as an EnvHub repo (`--env.hub_path=<user>/<repo>
--trust_remote_code=True`). The downloaded file falls back to importing `envhub` and
`shape_sorting` from the local install, which is unavoidable: the environment itself
lives in this repo and needs Isaac Lab, so the Hub copy saves no setup.
