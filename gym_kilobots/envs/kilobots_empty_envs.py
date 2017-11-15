from .kilobots_env import KilobotsEnv

import numpy as np
from scipy import stats

from ..lib.kilobot import PhototaxisKilobot
from ..lib.light import CircularGradientLight


class UniteKilobotsEnv(KilobotsEnv):
    world_size = world_width, world_height = 1., .5

    def __init__(self):
        # distribution for sampling swarm position
        self._kilobots_spawn_distribution = stats.uniform(loc=(-.95, -.7), scale=(1.8, 1.4))
        # distribution for sampling the pushing object
        self._light_spawn_distribution = stats.uniform(loc=(-.95, -.7), scale=(1.8, 1.4))

        self._num_kilobots = 20

        super().__init__()

    def _configure_environment(self):
        # create light
        self._lights = [CircularGradientLight(position=self._light_spawn_distribution.rvs(), radius=.2)]

        # create kilobots
        self._kilobots = [PhototaxisKilobot(self.world, position=self._kilobots_spawn_distribution.rvs(),
                                            lights=self._lights) for _ in range(self._num_kilobots)]

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