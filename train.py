#!/usr/bin/env python3
from unityagents import UnityEnvironment

import agents
import config as cfg
import memory as mem
from environment import Environment
from training import train

env_unity = UnityEnvironment(file_name=cfg.PATH_TO_CC, worker_id=1, no_graphics=True)
env = Environment(env_unity)

hpara = {
    "tau": 1e-3,
    "gamma": 0.99,
    "lr_actor": 1e-3,
    "lr_critic": 1e-3,
    "weight_decay_critic": 0.0,
    "noise_sigma": 0.4,
    "noise_theta": 0.15,
    "learn_every_step": 20,
    "learn_steps": 10,
    "noise_level_start": 1.0,
    "noise_level_range": 0.0,
    "noise_level_decay": 1.0,
}

agent = agents.DDPG(
    env.state_space_size,
    env.action_space_size,
    seed=123,
    hpara=hpara,
    memory_type=mem.Types.VANILLA_REPLAY,
)


train(
    num_episodes=300,
    early_end_sliding_score=30.0,
    early_end_num_episodes=100,
    env=env,
    agent=agent,
)

env.close()
