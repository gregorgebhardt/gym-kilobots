import time

import numpy as np
from scipy import stats

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from Box2D import b2World, b2ChainShape, b2Vec2

# import os, signal

from ..lib.body import Body, CornerQuad
from ..lib.kilobot import Kilobot, PhototaxisKilobot
from ..lib.light import Light, CircularGradientLight, GradientLight


class KilobotsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    world_size = world_width, world_height = 2., 1.5

    screen_size = screen_width, screen_height = 1200, 900

    sim_step = 1. / 60

    def __init__(self):
        # create the Kilobots world in Box2D
        self.world = b2World(gravity=(0, 0), doSleep=True)
        table = self.world.CreateStaticBody(position=(.0, .0))
        table.CreateFixture(
            shape=b2ChainShape(vertices=[(-self.world_width / 2, self.world_height / 2),
                                         (-self.world_width / 2, -self.world_height / 2),
                                         (self.world_width / 2, -self.world_height / 2),
                                         (self.world_width / 2, self.world_height / 2)]))

        # add objects
        self._objects: [Body] = []
        # add kilobots
        self._kilobots: [Kilobot] = []
        # add light
        self._lights: [Light] = []

        self.action_space = None
        self.observation_space = None

        self._configure_environment()

        if len(self._lights) > 0:
            self.action_space = spaces.Tuple(list(l.action_space for l in self._lights if l.action_space is not None))
        else:
            self.action_space = spaces.Tuple(())

        self._screen = None

    def _configure_environment(self):
        pass

    def _destroy(self):
        del self._objects[:]
        del self._kilobots[:]
        del self._lights[:]

    def _reset(self):
        self._destroy()
        self._configure_environment()

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        # step light
        for l, a in zip(self._lights, action):
            l.step(a)

        # step kilobots
        for k in self._kilobots:
            k.step(self.sim_step)

        # step world
        self.world.Step(self.sim_step, 60, 20)
        self.world.ClearForces()

    def _render(self, mode='human', close=False):
        if close:
            if self._screen is not None:
                # self._screen.close()
                self._screen = None
            return

        from ..lib import kb_rendering
        if self._screen is None:
            self._screen = kb_rendering.KilobotsViewer(self.screen_width, self.screen_height)
            self._screen.set_bounds(-1.04, 1.04, -.78, .78)

        # render table
        self._screen.draw_polygon([(-1.04, .78), (-1.04, -.78), (1.04, -.78), (1.04, .78)], color=(75, 75, 75))
        self._screen.draw_polygon([(-1., .75), (-1., -.75), (1., -.75), (1., .75)], color=(255, 255, 255))
        self._screen.draw_polyline([(-1., .75), (-1., -.75), (1., -.75), (1., .75), (-1., .75)], width=.005)

        # render light
        for l in self._lights:
            l.draw(self._screen)

        # render objects
        for o in self._objects:
            o.draw(self._screen)

        # render kilobots
        for kb in self._kilobots:
            kb.draw(self._screen)

        self._screen.render()


class QuadAssemblyKilobotsEnv(KilobotsEnv):
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
            CornerQuad(world=self.world, width=.15, height=.15, position=(.605, .605), rotation=-np.pi/2),
            CornerQuad(world=self.world, width=.15, height=.15, position=(.605, .45), rotation=-np.pi),

            CornerQuad(world=self.world, width=.15, height=.15, position=obj_spawn_location, rotation=-np.pi / 2)
        ]

        # create light
        self._lights = [CircularGradientLight(position=(.0, .1))]  # swarm_spawn_location
        # self._lights = [GradientLight(np.array([0, .75]), np.array([0, -.75]))]

        # create kilobots
        self._kilobots = [PhototaxisKilobot(self.world, position=(.0, .0), lights=self._lights)]
