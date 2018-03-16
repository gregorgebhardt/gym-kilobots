from .kilobots_env import KilobotsEnv

import numpy as np
from scipy import stats

from ..lib.body import CornerQuad
from ..lib.kilobot import PhototaxisKilobot, SimplePhototaxisKilobot
from ..lib.light import CircularGradientLight, GradientLight


class QuadPushingEnv(KilobotsEnv):
    world_size = world_width, world_height = 1., .5

    def __init__(self):
        # distribution for sampling swarm position
        self._swarm_spawn_distribution = stats.uniform(loc=(-.95, -.7), scale=(.9, 1.4))
        # distribution for sampling the pushing object
        self._obj_spawn_distribution = stats.uniform(loc=(.05, -.7), scale=(.9, .65))

        super().__init__()

    def _configure_environment(self):
        # sample swarm spawn location
        swarm_spawn_location = self._swarm_spawn_distribution.rvs()

        # sample object location
        obj_spawn_location = self._obj_spawn_distribution.rvs()

        # create objects
        self._objects = [
            CornerQuad(world=self.world, width=.15, height=.15, position=(.45, .605)),
            CornerQuad(world=self.world, width=.15, height=.15, position=(.605, .605), orientation=-np.pi / 2),
            CornerQuad(world=self.world, width=.15, height=.15, position=(.605, .45), orientation=-np.pi),
            CornerQuad(world=self.world, width=.15, height=.15, position=obj_spawn_location, orientation=-np.pi / 2)
        ]

        # create light
        self._light = CircularGradientLight(position=swarm_spawn_location)  # swarm_spawn_location
        self.action_space = self._light.action_space

        # create kilobots
        self._kilobots = [PhototaxisKilobot(self.world, position=swarm_spawn_location + (.0, .0),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.03, .0),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.0, .03),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (-.03, .0),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.0, -.03),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.03, .03),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (-.03, .03),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (-.03, -.03),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.03, -.03),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.06, .0),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.0, .06),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (-.06, .0),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.0, -.06),
                                            light=self._light)
                          ]

    # TODO implement has_finished function
    def has_finished(self, state, action):
        return False

    # TODO implement reward function
    def get_reward(self, state, action, new_state):
        # compute reward based on task and swarm state
        return 1.

    # info function
    def get_info(self, state, action):
        return None


class QuadAssemblyKilobotsEnv(KilobotsEnv):
    world_size = world_width, world_height = 1., 1.
    screen_size = screen_width, screen_height = 800, 800

    def __init__(self):
        # distribution for sampling swarm position
        self._swarm_spawn_distribution = stats.uniform(loc=(-.45, -.45), scale=(.9, .9))
        # distribution for sampling the pushing object
        self._obj_spawn_distribution = stats.uniform(loc=(-.45, -.45), scale=(.4, .4))

        super().__init__()

    def _configure_environment(self):
        # sample swarm spawn location
        swarm_spawn_location = self._swarm_spawn_distribution.rvs()

        # sample object location
        obj_spawn_location = self._obj_spawn_distribution.rvs()

        # create objects
        self._objects = [
            CornerQuad(world=self.world, width=.15, height=.15, position=(.15, .305)),
            CornerQuad(world=self.world, width=.15, height=.15, position=(.305, .305), orientation=-np.pi / 2),
            CornerQuad(world=self.world, width=.15, height=.15, position=(.305, .15), orientation=-np.pi),
            CornerQuad(world=self.world, width=.15, height=.15, position=obj_spawn_location, orientation=-np.pi / 2)
        ]

        # create light
        self._light = CircularGradientLight(position=swarm_spawn_location)  # swarm_spawn_location
        self.action_space = self._light.action_space

        # create kilobots
        self._kilobots = [PhototaxisKilobot(self.world, position=swarm_spawn_location + (.0, .0),
                                                  light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.03, .0),
                                                  light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.0, .03),
                                                  light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (-.03, .0),
                                                  light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.0, -.03),
                                                  light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.0, .0),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.03, .0),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.0, .03),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (-.03, .0),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.0, -.03),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.0, .0),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.03, .0),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.0, .03),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (-.03, .0),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.0, -.03),
                                            light=self._light)
                          ]

    # TODO implement has_finished function
    def has_finished(self, state, action):
        return False

    # TODO implement reward function
    def get_reward(self, state, action, new_state):
        # compute reward based on task and swarm state
        return 1.

    # info function
    def get_info(self, state, action):
        return None


class GradientLightEnv(KilobotsEnv):
    world_size = world_width, world_height = 1., 1.

    def __init__(self):
        super().__init__()

    def _configure_environment(self):
        # create objects
        # self._objects = []

        # create light
        self._light = GradientLight(angle=-.25*np.pi)
        self.action_space = self._light.action_space

        # create kilobots
        self._kilobots = []

        for i in range(20):
            kilobot_position = np.random.rand(2) - .5
            self._kilobots.append(SimplePhototaxisKilobot(self.world, position=kilobot_position, light=self._light))

    # TODO implement has_finished function
    def has_finished(self, state, action):
        return False

    # TODO implement reward function
    def get_reward(self, state, action, new_state):
        # compute reward based on task and swarm state
        return 1.

    # info function
    def get_info(self, state, action):
        return None