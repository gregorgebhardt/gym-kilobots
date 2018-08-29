import gym
from gym import error, spaces, utils

import sys
import logging

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
    __viz_steps_per_second = 20
    __steps_per_action = 10

    # in addition to action space and observation space, we also define the state space
    state_space = None

    def __new__(cls, *args, **kwargs):
        cls.sim_steps_per_second = cls.__sim_steps_per_second
        cls.sim_step = 1. / cls.__sim_steps_per_second
        cls.world_x_range = -cls.world_width / 2, cls.world_width / 2
        cls.world_y_range = -cls.world_height / 2, cls.world_height / 2
        cls.world_bounds = (np.array([-cls.world_width / 2, -cls.world_height / 2]),
                            np.array([cls.world_width / 2, cls.world_height / 2]))

        return super(KilobotsEnv, cls).__new__(cls)

    def __init__(self):
        self.__sim_steps = 0

        # create the Kilobots world in Box2D
        self.world = b2World(gravity=(0, 0), doSleep=True)
        self.table = self.world.CreateStaticBody(position=(.0, .0))
        self.table.CreateFixture(
            shape=b2ChainShape(vertices=[(self.world_x_range[0], self.world_y_range[1]),
                                         (self.world_x_range[0], self.world_y_range[0]),
                                         (self.world_x_range[1], self.world_y_range[0]),
                                         (self.world_x_range[1], self.world_y_range[1])]))
        self.table.fixtures[0].shape.radius = .0001

        # add kilobots
        self._kilobots: [Kilobot] = []
        # add objects
        self._objects: [Body] = []
        # add light
        self._light: Light = None

        self.__seed = 0

        self._screen = None

        self._configure_environment()

    @property
    def _sim_steps(self):
        return self.__sim_steps

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

    def get_observation(self):
        return self.get_state()

    @abc.abstractmethod
    def get_reward(self, state, action, new_state):
        raise NotImplementedError

    def has_finished(self, state, action):
        return False

    def get_info(self, state, action):
        return ""

    def destroy(self):
        del self._objects[:]
        del self._kilobots[:]
        del self._light
        self._light = None
        if self._screen is not None:
            del self._screen
            self._screen = None

    def close(self):
        self.destroy()

    def seed(self, seed=None):
        if seed is not None:
            self.__seed = seed
        return [self.__seed]

    def reset(self):
        self.destroy()
        self._configure_environment()
        self.__sim_steps = 0

        return self.get_observation()

    def step(self, action: np.ndarray):
        # if self.action_space and action is not None:
        #     assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        # state before action is applied
        old_state = self.get_state()

        for i in range(self.__steps_per_action):
            # step light
            if action is not None:
                if self._light.interpolate_actions:
                    if self._light.relative_actions:
                        self._light.step(action / self.__steps_per_action)
                    else:
                        light_state = self._light.get_state()
                        rel_sub_action = (action - light_state) * i / self.__steps_per_action
                        self._light.step(light_state + rel_sub_action)
                else:
                    if i == 0:
                        self._light.step(action)
                    else:
                        self._light.step(None)

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

        # observation
        observation = self.get_observation()

        # reward
        reward = self.get_reward(old_state, action, new_state)

        # done
        done = self.has_finished(new_state, action)

        # info
        info = self.get_info(new_state, action)

        return observation, reward, done, info

    def _step_world(self):
        self.world.Step(self.sim_step, self.__sim_velocity_iterations, self.__sim_position_iterations)
        self.world.ClearForces()

    def render(self, mode='human'):
        # if close:
        #     if self._screen is not None:
        #         self._screen.close()
        #         self._screen = None
        #     return

        if self.__sim_steps % self.__sim_steps_per_second // self.__viz_steps_per_second:
            return

        from ..lib import kb_rendering
        if self._screen is None:
            caption = self.spec.id if self.spec else ""
            self._screen = kb_rendering.KilobotsViewer(self.screen_width, self.screen_height, caption=caption)
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

        # allow to draw on table
        self._draw_on_table(self._screen)

        # render light
        self._light.draw(self._screen)

        # render objects
        for o in self._objects:
            o.draw(self._screen)

        # render kilobots
        for kb in self._kilobots:
            kb.draw(self._screen)

        # allow to draw on top
        self._draw_on_top(self._screen)

        self._screen.render()

    def get_index(self):
        kilobot_index = pd.MultiIndex.from_product([range(len(self._kilobots)), ['x', 'y', 'theta']],
                                                   names=['idx', 'dim'])
        objects_index = pd.MultiIndex.from_product([range(len(self._objects)), ['x', 'y', 'theta']],
                                                   names=['idx', 'dim'])
        light_index = self._light.get_index()

        return kilobot_index, objects_index, light_index

    def get_objects(self) -> [Body]:
        return self._objects

    def get_kilobots(self) -> [Kilobot]:
        return self._kilobots

    def get_light(self) -> Light:
        return self._light

    def _draw_on_table(self, screen):
        pass

    def _draw_on_top(self, screen):
        pass
