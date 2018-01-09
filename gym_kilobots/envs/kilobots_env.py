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

    __sim_steps_per_second = 10
    __sim_velocity_iterations = 10
    __sim_position_iterations = 10
    __sim_steps = 0
    __viz_steps_per_second = 20
    __steps_per_action = 10

    @property
    def sim_steps_per_second(self):
        return self.__sim_steps_per_second

    @property
    def sim_step(self):
        return 1. / self.__sim_steps_per_second

    @property
    def world_x_range(self):
        return -self.world_width / 2, self.world_width / 2

    @property
    def world_y_range(self):
        return -self.world_height / 2, self.world_height / 2

    @property
    def world_bounds(self):
        return (np.array([-self.world_width / 2, -self.world_height / 2]),
                np.array([self.world_width / 2, self.world_height / 2]))

    def __init__(self):
        # create the Kilobots world in Box2D
        self.world = b2World(gravity=(0, 0), doSleep=True)
        table = self.world.CreateStaticBody(position=(.0, .0))
        table.CreateFixture(
            shape=b2ChainShape(vertices=[(self.world_x_range[0], self.world_y_range[1]),
                                         (self.world_x_range[0], self.world_y_range[0]),
                                         (self.world_x_range[1], self.world_y_range[0]),
                                         (self.world_x_range[1], self.world_y_range[1])]))
        table.fixtures[0].shape.radius = .001

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

        self.__seed = None

    def _add_kilobot(self, kilobot: Kilobot):
        self._kilobots.append(kilobot)

    def _add_object(self, body: Body):
        self._objects.append(body)

    @abc.abstractmethod
    def _configure_environment(self):
        raise NotImplementedError

    def get_state(self):
        return {'kilobots': np.array([k.get_state() for k in self._kilobots]),
                'objects': np.array([o.get_state() for o in self._objects]),
                'light': self._light.get_state()}

    @abc.abstractmethod
    def _reward(self, state, action, new_state):
        raise NotImplementedError

    def _has_finished(self, state, action):
        return False

    def _get_info(self, state, action):
        return ""

    def _destroy(self):
        del self._objects[:]
        del self._kilobots[:]
        del self._light
        self._light = None
        if self._screen is not None:
            del self._screen
            self._screen = None

    def _close(self):
        self._destroy()

    def _seed(self, seed=None):
        if seed is not None:
            self.__seed = seed
        return self.__seed

    def _reset(self):
        self._destroy()
        self._configure_environment()
        self.__sim_steps = 0

    def _step(self, action):
        if self.action_space:
            assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        state = self.get_state()

        for i in range(self.__steps_per_action):
            # step light
            if i == 0:
                self._light.step(action)
            else:
                self._light.step(np.zeros(2))

            # step kilobots
            for k in self._kilobots:
                k.step(self.sim_step)

            # step world
            self.world.Step(self.sim_step, self.__sim_velocity_iterations, self.__sim_position_iterations)
            self.world.ClearForces()

            self.__sim_steps += 1

            if self._screen is not None:
                self.render()

        # state
        new_state = self.get_state()

        # reward
        reward = self._reward(state, action, new_state)

        # done
        done = self._has_finished(new_state, action)

        # info
        info = self._get_info(new_state, action)

        return state, reward, done, info

    def _step_world(self):
        self.world.Step(self.sim_step, self.__sim_velocity_iterations, self.__sim_position_iterations)
        self.world.ClearForces()

    def _render(self, mode='human', close=False):
        if close:
            if self._screen is not None:
                self._screen.close()
                self._screen = None
            return

        if self.__sim_steps % self.__sim_steps_per_second // self.__viz_steps_per_second:
            return

        from ..lib import kb_rendering
        if self._screen is None:
            self._screen = kb_rendering.KilobotsViewer(self.screen_width, self.screen_height, caption=self.spec.id)
            world_min, world_max = self.world_bounds
            self._screen.set_bounds(world_min[0], world_max[0], world_min[1], world_max[1])
        elif self._screen.close_requested():
            self._screen.close()
            self._screen = None
            # TODO how to handle this event?

        # render table
        x_min, x_max = self.world_x_range
        y_min, y_max = self.world_y_range
        self._screen.draw_polygon([(x_min, y_max), (x_min, y_min), (x_max, y_min), (x_max, y_max)],
                                  color=(255, 255, 255))
        self._screen.draw_polyline([(x_min, y_max), (x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)],
                                   width=.003)

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
