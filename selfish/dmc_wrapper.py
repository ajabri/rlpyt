from gym import core, spaces
from dm_control import suite
from dm_control.rl.control import specs
from gym.utils import seeding
import gym
from gym.spaces import Tuple

from rlpyt.spaces.composite import Composite
# from dm_control2gym.viewer import DmControlViewer
import numpy as np
import sys

import pyglet
import numpy as np

class DmControlViewer:
    def __init__(self, width, height, depth=False):
        self.window = pyglet.window.Window(width=width, height=height, display=None)
        self.width = width
        self.height = height

        self.depth = depth

        if depth:
            self.format = 'RGB'
            self.pitch = self.width * -3
        else:
            self.format = 'RGB'
            self.pitch = self.width * -3

    def update(self, pixel):
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        if self.depth:
            pixel = np.dstack([pixel.astype(np.uint8)] * 3)
        pyglet.image.ImageData(self.width, self.height, self.format, pixel.tobytes(), pitch=self.pitch).blit(0, 0)
        self.window.flip()

    def close(self):
        self.window.close()


class DmcDiscrete(gym.spaces.Discrete):
    def __init__(self, _minimum, _maximum):
        super().__init__(_maximum - _minimum)
        self.offset = _minimum


class Batch(Tuple):
    """
    A batch (i.e., product) of identical simpler spaces
    Example usage:
    self.observation_space = spaces.Batch((spaces.Discrete(3) for _ in range(N)))
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = np.asarray(self.sample()).shape

    def sample(self):
        return [space.sample() for space in self.spaces]
        # return np.concatenate([space.sample() for space in self.spaces])

def convertSpec2Space(spec, clip_inf=False):
    # if isinstance(spec, list):
    #     return Tuple([convertSpec2Space(s) for s in spec])
    if isinstance(spec, list):
        return Batch([convertSpec2Space(s) for s in spec])

    if spec.dtype == np.int:
        # Discrete
        return DmcDiscrete(spec.minimum, spec.maximum)
    else:
        # Box
        if type(spec) is specs.Array:
            return spaces.Box(-np.inf, np.inf, shape=spec.shape)
        elif type(spec) is specs.BoundedArray:
            _min = spec.minimum
            _max = spec.maximum
            if clip_inf:
                _min = np.clip(spec.minimum, -sys.float_info.max, sys.float_info.max)
                _max = np.clip(spec.maximum, -sys.float_info.max, sys.float_info.max)

            if np.isscalar(_min) and np.isscalar(_max):
                # same min and max for every element
                return spaces.Box(_min, _max, shape=spec.shape)
            else:
                # different min and max for every element
                return spaces.Box(_min + np.zeros(spec.shape),
                                  _max + np.zeros(spec.shape))
        else:
            raise ValueError('Unknown spec!')

def convertOrderedDict2Space(odict):
    if isinstance(odict, list):
        return Batch([convertOrderedDict2Space(s) for s in odict])

    if len(odict.keys()) == 1:
        # no concatenation
        return convertSpec2Space(list(odict.values())[0])
    else:
        # concatentation
        numdim = sum([np.int(np.prod(odict[key].shape)) for key in odict])
        return spaces.Box(-np.inf, np.inf, shape=(numdim,))


def convertObservation(spec_obs):
    if isinstance(spec_obs, list):
        return np.stack([convertObservation(s) for s in spec_obs])

    if len(spec_obs.keys()) == 1:
        # no concatenation
        return list(spec_obs.values())[0]
    else:
        # concatentation
        numdim = sum([np.int(np.prod(spec_obs[key].shape)) for key in spec_obs])
        space_obs = np.zeros((numdim,))
        i = 0
        for key in spec_obs:
            space_obs[i:i+np.prod(spec_obs[key].shape)] = spec_obs[key].ravel()
            i += np.prod(spec_obs[key].shape)
        return space_obs


class DmControlWrapper(core.Env):

    def __init__(self, domain_name, task_name, task_kwargs=None, visualize_reward=False, render_mode_list=None, env=None):
        if env is None:
            self.dmcenv = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs=task_kwargs,
                                 visualize_reward=visualize_reward)
        else:
            self.dmcenv = env

        # convert spec to space
        self.action_space = convertSpec2Space(self.dmcenv.action_spec(), clip_inf=True)
        self.observation_space = convertOrderedDict2Space(self.dmcenv.observation_spec())

        if render_mode_list is not None:
            self.metadata['render.modes'] = list(render_mode_list.keys())
            self.viewer = {key:None for key in render_mode_list.keys()}
        else:
            self.metadata['render.modes'] = []

        self.render_mode_list = render_mode_list

        # set seed
        self.seed()

    def getObservation(self):
        return convertObservation(self.timestep.observation)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.timestep = self.dmcenv.reset()
        return self.getObservation()

    def step(self, a):

        if type(self.action_space) == DmcDiscrete:
            a += self.action_space.offset
        self.timestep = self.dmcenv.step(a)

        return self.getObservation(), np.asarray(self.timestep.reward), self.timestep.last(), {}


    def render(self, mode='human', close=False):

        self.pixels = self.dmcenv.physics.render(**self.render_mode_list[mode]['render_kwargs'])
        if close:
            if self.viewer[mode] is not None:
                self.get_viewer(mode).close()
                self.viewer[mode] = None
            return
        elif self.render_mode_list[mode]['show']:
            self.get_viewer(mode).update(self.pixels)


        if self.render_mode_list[mode]['return_pixel']:

            return self.pixels

    def get_viewer(self, mode):
        if self.viewer[mode] is None:
            self.viewer[mode] = DmControlViewer(self.pixels.shape[1], self.pixels.shape[0], self.render_mode_list[mode]['render_kwargs']['depth'])
        return self.viewer[mode]
