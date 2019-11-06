import sys
import numpy as np

# Environments stuff
import gym
import dmc_wrapper as dmc2gym
from rlpyt.envs.gym import GymEnvWrapper

import sys
sys.path.append('../../multiagentworld')
import maw

from absl import logging
logging.set_verbosity(logging.ERROR)

# rlpyt stuff
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
# from rlpyt.envs.gym import make as gym_make
from rlpyt.algos.pg.ppo import PPO
from rlpyt.agents.pg.mujoco import MujocoFfAgent
from rlpyt.agents.pg.multiagent import MultiAgentGaussianPgAgent, MultiFfAgent
from rlpyt.runners.minibatch_rl import MinibatchRl, MinibatchRlEval
from rlpyt.utils.logging.context import logger_context, get_log_dir

from my_config import configs
from rlpyt.utils.launching.affinity import get_n_run_slots, prepend_run_slot, affinity_from_code, encode_affinity

import utils

##################### ARGS #####################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", default="test_maw", type=str)
parser.add_argument("--task-id", default="MaxHeightTask", type=str)
parser.add_argument("--team-size", default=1, type=int)
parser.add_argument("--time-limit", default=10, type=int)

args = parser.parse_args()
args.exp_name = "%s_%s" % (args.exp_name, '-'.join([
    "%s:%s" % (k,getattr(args, k)) for k in ['task_id', 'team_size']
]))

##################### Env constructor #####################

def make_env():
    # Load the 2-vs-2 soccer environment with episodes of 10 seconds:
    dm_env = maw.load(team_size=1, time_limit=10., task_id='MaxHeightTask')
    #dm_env = dm_soccer.load(team_size=2, time_limit=10.)
    env = GymEnvWrapper(dmc2gym.DmControlWrapper('', '', env=dm_env))
    return env

##################### Visualization function #####################
import maw.utils

env = make_env().env.dmcenv
arena = env.task.arena
terrain = maw.pitch.make_terrain(arena._res, arena._size)

# Get the part of state that we care about (pose)
obs_indices = [(0, (0,0))]
for k,v in env.observation_spec()[0].items():
    print(k, v.shape)
    obs_indices.append((k, (v.shape[-1], v.shape[-1] + obs_indices[-1][-1][-1])))
obs_indices = dict(obs_indices)
idx = obs_indices['position']

def vis_trajs(samples, itr):
    ''' MiniBatchRL runner will call this function, feeding it samples '''
    poses = samples.env.observation[..., idx[1]:idx[1]+idx[0]-1]
    return maw.utils.dump_traj_vis(
        poses, terrain, arena._size, itr, outdir=get_log_dir("run_%s" % args.exp_name))

##################### Parallelization stuff #####################

affinity_code = encode_affinity(
    n_cpu_core=8,
    n_gpu=0,
    hyperthread_offset=2,
    n_socket=1,
    cpu_per_run=8,
)


def build_and_train(log_dir, run_ID, config_key):
    # affinity = affinity_from_code(run_slot_affinity_code)
    slot_affinity_code = prepend_run_slot(0, affinity_code)
    affinity = affinity_from_code(slot_affinity_code)

    config = configs[config_key]

    sampler = CpuSampler(
        EnvCls=make_env,
        env_kwargs={},
        CollectorCls=CpuResetCollector,
        **config["sampler"]
    )
    algo = PPO(optim_kwargs=config["optim"], **config["algo"])
    agent = MultiFfAgent(model_kwargs=config["model"], **config["agent"])
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        diag_fn=vis_trajs,
        **config["runner"])
    name = config["env"]["id"]

    with logger_context(log_dir, run_ID, name, config):
        runner.train()


build_and_train('', args.exp_name, "ppo_1M_cpu")
