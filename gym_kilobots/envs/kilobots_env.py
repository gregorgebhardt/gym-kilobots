import gym
from gym.utils import seeding
from gym import error, spaces, utils

import numpy as np
import pandas as pd

from Box2D import b2World, b2ChainShape

# import os, signal

from ..lib.body import Body
from ..lib.kilobot import Kilobot
from ..lib.light import Light

import abc


class KilobotsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    world_size = world_width, world_height = 2., 1.5
    screen_size = screen_width, screen_height = 1200, 900
    sim_step = 1. / 60

    @property
    def world_x_range(self):
        return -self.world_width / 2, self.world_width / 2

    @property
    def world_y_range(self):
        return -self.world_height / 2, self.world_height / 2

    def __init__(self):
        # create the Kilobots world in Box2D
        self.world = b2World(gravity=(0, 0), doSleep=True)
        table = self.world.CreateStaticBody(position=(.0, .0))
        table.CreateFixture(
            shape=b2ChainShape(vertices=[(self.world_x_range[0], self.world_y_range[1]),
                                         (self.world_x_range[0], self.world_y_range[0]),
                                         (self.world_x_range[1], self.world_y_range[0]),
                                         (self.world_x_range[1], self.world_y_range[1])]))

        # add kilobots
        self._kilobots: [Kilobot] = []
        # add objects
        self._objects: [Body] = []
        # add light
        self._light: Light = None

        self.action_space = None
        self.observation_space = None

        self._configure_environment()

        if self._light is not None:
            self.action_space = self._light.action_space
        else:
            self.action_space = None

        # construct observation space
        kb_low = np.array([[self.world_x_range[0], self.world_y_range[0], -np.inf]] * len(self._kilobots))
        kb_high = np.array([[self.world_x_range[1], self.world_y_range[1], np.inf]] * len(self._kilobots))
        kb_observation_space = spaces.Box(low=kb_low, high=kb_high)

        objects_low = np.array([[self.world_x_range[0], self.world_y_range[0], -np.inf]] * len(self._objects))
        objects_high = np.array([[self.world_x_range[1], self.world_y_range[1], np.inf]] * len(self._objects))
        objects_observation_space = spaces.Box(low=objects_low, high=objects_high)

        self.observation_space = spaces.Tuple([kb_observation_space, objects_observation_space,
                                               self._light.observation_space])

        self._screen = None

    def _add_kilobot(self, kilobot: Kilobot):
        self._kilobots.append(kilobot)

    def _add_object(self, object: Body):
        self._objects.append(object)

    @abc.abstractmethod
    def _configure_environment(self):
        raise NotImplementedError

    def _get_state(self):
        return {'kilobots': np.array([k.get_state() for k in self._kilobots]),
                'objects': np.array([o.get_state() for o in self._objects]),
                'light': self._light.get_state()}

    @abc.abstractmethod
    def _reward(self, state, action):
        raise NotImplementedError

    def _has_finished(self, state, action):
        return False

    def _get_info(self, state, action):
        return None

    def _destroy(self):
        del self._objects[:]
        del self._kilobots[:]
        del self._light

    def _reset(self):
        self._destroy()
        self._configure_environment()

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        # step light
        self._light.step(action)

        # step kilobots
        for k in self._kilobots:
            k.step(self.sim_step)

        # step world
        self.world.Step(self.sim_step, 60, 20)
        self.world.ClearForces()

        # state
        state = self._get_state()

        # reward
        reward = self._reward(state, action) or None

        # done
        done = self._has_finished(state, action) or None

        # info
        info = self._get_info(state, action) or None

        return state, reward, done, info

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
        self._light.draw(self._screen)

        # render objects
        for o in self._objects:
            o.draw(self._screen)

        # render kilobots
        for kb in self._kilobots:
            kb.draw(self._screen)

        self._screen.render()

    def get_index(self):
        kilobot_index = pd.MultiIndex.from_product([range(len(self._kilobots)), ['x', 'y', 'theta']],
                                                   names=['idx', 'dim'])
        objects_index = pd.MultiIndex.from_product([range(len(self._objects)), ['x', 'y', 'theta']],
                                                   names=['idx', 'dim'])
        light_index = self._light.get_index()

        return kilobot_index, objects_index, light_index