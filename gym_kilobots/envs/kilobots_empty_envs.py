from .kilobots_env import KilobotsEnv

import numpy as np
from scipy import stats

from ..lib.kilobot import *
from ..lib.light import *


class UniteKilobotsEnv(KilobotsEnv):
    world_size = world_width, world_height = 1., .5
    screen_size = screen_width, screen_height = 1000, 500

    def __init__(self):
        # distribution for sampling swarm position
        self._kilobots_spawn_distribution = stats.uniform(loc=(-.45, -.2), scale=(.9, .4))
        # distribution for sampling the pushing object
        self._light_spawn_distribution = stats.uniform(loc=(-.45, -.2), scale=(.9, .4))

        self._num_kilobots = 10

        super().__init__()

    def _configure_environment(self):
        # create light
        # self._light = CircularGradientLight(position=self._light_spawn_distribution.rvs())
        # self._light = GradientLight(gradient_start=np.array([0, -.25]), gradient_end=np.array([.0, .25]))
        self._light = SinglePositionLight(position=self._light_spawn_distribution.rvs())

        # create kilobots
        self._kilobots = [SimplePhototaxisKilobot(self.world, position=self._kilobots_spawn_distribution.rvs(),
                                                  light=self._light) for _ in range(self._num_kilobots)]

    def _has_finished(self, state, action):
        positions = state['kilobots'][:, :2]
        if positions.std(axis=0).max() < .05:
            return True
        return False

    def _reward(self, state, action):
        # compute reward based on task and swarm state
        positions = state['kilobots'][:, :2]
        return -1 * positions.std(axis=0).max()

    # info function
    def _get_info(self, state, action):
        return None