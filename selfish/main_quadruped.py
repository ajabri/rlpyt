import sys
import numpy as np

# Environments stuff
import gym
from dm_control import suite
from dm_control.locomotion import soccer as dm_soccer
import dmc_wrapper as dmc2gym


# rlpyt stuff
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
# from rlpyt.envs.gym import make as gym_make

from rlpyt.algos.pg.ppo import PPO
from rlpyt.agents.pg.mujoco import MujocoFfAgent
from rlpyt.agents.pg.multiagent import MultiAgentGaussianPgAgent, MultiFfAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.variant import load_variant, update_config

from rlpyt.envs.gym import GymEnvWrapper

# from rlpyt.experiments.configs.mujoco.pg.mujoco_ppo import configs
from my_config import configs
from rlpyt.utils.launching.affinity import get_n_run_slots, prepend_run_slot, affinity_from_code, encode_affinity

import utils

def make_env():
    # Load the 2-vs-2 soccer environment with episodes of 10 seconds:
    # dm_env = dm_soccer.load(team_size=2, time_limit=10.)
    dm_env = suite.load(domain_name="quadruped", task_name="escape")
    env = GymEnvWrapper(dmc2gym.DmControlWrapper('', '', env=dm_env))
    return env

affinity_code = encode_affinity(
    n_cpu_core=10,
    n_gpu=0,
    hyperthread_offset=2,
    n_socket=1,
    cpu_per_run=10,
)


def build_and_train(slot_affinity_code, log_dir, run_ID, config_key):
    if slot_affinity_code is 'None':
        # affinity = affinity_from_code(run_slot_affinity_code)
        slot_affinity_code = prepend_run_slot(0, affinity_code)
        affinity = affinity_from_code(slot_affinity_code)
    else:
        affinity = affinity_from_code(slot_affinity_code)

    config = configs[config_key]

    # load variant of experiment (there may not be a variant, though)
    variant = load_variant(log_dir)
    config = update_config(config, variant)

    sampler = CpuSampler(
        EnvCls=make_env,
        env_kwargs={},
        CollectorCls=CpuResetCollector,
        **config["sampler"]
    )
    algo = PPO(optim_kwargs=config["optim"], **config["algo"])
    agent = MujocoFfAgent(model_kwargs=config["model"], **config["agent"])
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config["runner"]
    )
    name = config["env"]["id"]
    with logger_context(log_dir, run_ID, name, config):
        runner.train()

if __name__ == "__main__":
    build_and_train(*sys.argv[1:])
    
# build_and_train('logs/', 'test', "ppo_1M_cpu")
    