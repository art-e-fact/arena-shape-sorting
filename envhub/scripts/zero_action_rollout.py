"""Smoke-test the EnvHub wiring: build the env and step zero actions, no policy involved.

    PYTHONPATH=. /isaac-sim/python.sh envhub/scripts/zero_action_rollout.py \
        --env.discover_packages_path=envhub \
        --env.type=shape_sorting_arena \
        --env.visualizer=kit
"""

import logging
from pprint import pformat

import torch
import tqdm

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env


@parser.wrap()
def main(cfg: EvalPipelineConfig):
    logging.info(pformat(cfg.env))

    env = next(iter(make_env(cfg.env, n_envs=cfg.eval.batch_size).values()))[0]
    try:
        obs, _ = env.reset()
        print(f"state terms: {list(obs['policy'])}")
        print(f"camera terms: {list(obs.get('camera_obs', {}))}")
        print(f"task: {env.task!r}, max steps: {env.call('_max_episode_steps')[0]}")

        actions = torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device)
        for _ in tqdm.tqdm(range(100), desc="zero actions"):
            with torch.inference_mode():
                env.step(actions)
    finally:
        env.close()


if __name__ == "__main__":
    main()
