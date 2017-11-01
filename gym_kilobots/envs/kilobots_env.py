import gym
from gym import error, spaces, utils
from gym.utils import seeding

from Box2D import b2World, b2ChainShape

import os, signal

from ..lib.body import Body, CornerQuad
from ..lib.kilobot import Kilobot
from ..lib.light import Light, CircularGradientLight


class KilobotsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    WIDTH = 2.
    HEIGHT = 1.5

    def __init__(self):
        # create the Kilobots world in Box2D
        self.world = b2World(gravity=(0, 0), doSleep=True)
        table = self.world.CreateStaticBody(position=(.0, .0))
        table.CreateFixture(
            shape=b2ChainShape(vertices=[(-self.WIDTH / 2, self.HEIGHT / 2), (-self.WIDTH / 2, -self.HEIGHT / 2),
                                               (self.WIDTH / 2, -self.HEIGHT / 2), (self.WIDTH / 2, self.HEIGHT / 2)]))

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
            self.action_space = spaces.Tuple((l.action_space for l in self._lights))

        self._viewer = None

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
        pass

    def _render(self, mode='human', close=False):
        if close:
            if self._viewer is not None:
                self._viewer.close()
                self._viewer = None
            return

        from gym.envs.classic_control import rendering
        if self._viewer is None:
            self._viewer = rendering.Viewer(800, 600)
            self._viewer.set_bounds(-1.04, 1.04, -.78, .78)

        # render table
        self._viewer.draw_polygon([(-1.04, .78), (-1.04, -.78), (1.04, -.78), (1.04, .78)], color=(.3, .3, .3))
        self._viewer.draw_polygon([(-1., .75), (-1., -.75), (1., -.75), (1., .75)], color=(1., 1., 1.))
        self._viewer.draw_polyline([(-1., .75), (-1., -.75), (1., -.75), (1., .75), (-1., .75)], linewidth=.01)

        # render objects
        for o in self._objects:
            o.draw(self._viewer)

        # render kilobots
        for kb in self._kilobots:
            kb.draw(self._viewer)

        # render light
        for l in self._lights:
            l.draw(self._viewer)

        self._viewer.render()


# class CircularLightGradientKilobotsEnv(KilobotsEnv):
#     def __init__(self):
#         super(CircularLightGradientKilobotsEnv, self).__init__()
#
#         self._objects = [CornerQuad(position=)]
