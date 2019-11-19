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
parser.add_argument("--agent-type", default="BoxHead", type=str)
parser.add_argument("--team-size", default=1, type=int)
parser.add_argument("--time-limit", default=20., type=float)
parser.add_argument("--test", default=False, action="store_true")
parser.add_argument("--random-spawn", default=False, action="store_true")
parser.add_argument("--no-hfield", default=False, action="store_true")
parser.add_argument("--reload-path", default="", type=str)

args = parser.parse_args()
args.exp_name = "%s_%s" % (args.exp_name, '-'.join(
    ["%s=%s" % (k, getattr(args, k)) for k in ['task_id', 'team_size', 'agent_type']]))

##################### Env constructor #####################


def make_env():
    # Load the 2-vs-2 soccer environment with episodes of 10 seconds:
    dm_env = maw.load(
        team_size=args.team_size,
        time_limit=args.time_limit,
        terrain=not args.no_hfield,
        agent_type=args.agent_type,
        deterministic_spawn=not args.random_spawn,
        raise_exception_on_physics_error=False,
        task_id=args.task_id)
    #dm_env = dm_soccer.load(team_size=2, time_limit=10.)
    env = GymEnvWrapper(dmc2gym.DmControlWrapper('', '', env=dm_env))
    return env


##################### Visualization function #####################
import maw.utils

env = make_env().env.dmcenv
arena = env.task.arena
terrain = maw.pitch.make_terrain(arena._res, arena._size)

# Get the part of state that we care about (pose)
obs_indices = [(0, (0, 0))]
for k, v in env.observation_spec()[0].items():
    print(k, v.shape)
    obs_indices.append((k, (v.shape[-1],
                            v.shape[-1] + obs_indices[-1][-1][-1])))
obs_indices = dict(obs_indices)


def vis_trajs(samples, itr):
    ''' MiniBatchRL runner will call this function, feeding it samples '''
    return
    # idx = obs_indices['position']
    # poses = samples.env.observation[..., idx[1]:idx[1] + idx[0] - 1]
    # return maw.utils.dump_traj_vis(
    #     poses,
    #     terrain,
    #     arena._size,
    #     itr,
    #     outdir=get_log_dir("run_%s" % args.exp_name))


##################### Parallelization stuff #####################

affinity_code = encode_affinity(
    n_cpu_core=12,
    n_gpu=0,
    hyperthread_offset=1,
    n_socket=1,
    cpu_per_run=12,
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
        **config["sampler"])
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

    with logger_context(log_dir, run_ID, name, config, snapshot_mode='last'):
        runner.train()


def build_and_test(model_path, config_key):
    import dmc_wrapper
    from dm_control import viewer
    from rlpyt.utils.buffer import buffer_from_example, torchify_buffer, numpify_buffer
    import torch

    config = configs[config_key]

    reloaded = torch.load(model_path) if len(model_path) > 0 else None
    # import pdb; pdb.set_trace()
    agent = MultiFfAgent(
        model_kwargs=config["model"],
        initial_model_state_dict=reloaded['agent_state_dict'],
        **config["agent"])

    dm_env = maw.load(
        team_size=args.team_size,
        time_limit=args.time_limit,
        terrain=not args.no_hfield,
        agent_type=args.agent_type,
        deterministic_spawn=not args.random_spawn,
        raise_exception_on_physics_error=False,
        task_id=args.task_id)

    env = GymEnvWrapper(dmc2gym.DmControlWrapper('', '', env=dm_env))

    agent.initialize(env.spaces)
    agent.reset()
    # agent.eval_mode(0)

    prev_action = env.action_space.null_value()

    def get_prev_action():
        return prev_action

    def policy(time_step):
        obs = dmc_wrapper.convertObservation(time_step.observation)
        reward = time_step.reward
        reward = np.asarray(reward) if reward is not None else reward

        obs_pyt, act_pyt, rew_pyt = torchify_buffer((obs, get_prev_action(),
                                                     reward))
        # obs_pyt, rew_pyt = torchify_buffer((obs, reward))

        act_pyt, agent_info = agent.step(obs_pyt.float(), act_pyt, rew_pyt)
        # prev_action = act_pyt

        return act_pyt

    viewer.launch(dm_env, policy=policy)


if args.test:
    build_and_test(args.reload_path, "ppo_1M_cpu")
else:
    build_and_train("", args.exp_name, "ppo_1M_cpu")
