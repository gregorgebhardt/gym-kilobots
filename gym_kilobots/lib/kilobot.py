import numpy as np
import Box2D

from .body import Circle

from gym.envs.classic_control import rendering

import math


class Kilobot(Circle):
    # all parameters in real world units
    _radius = 0.0165  # meters

    _max_linear_velocity = 0.01  # meters / s
    _max_angular_velocity = 0.1 * math.pi  # radians / s

    _density = 1.0
    _friction = 0.2
    _restitution = 0.0

    _linear_damping = 0.8
    _angular_damping = 0.8

    """
        scale_real_to_sim: scale factor to go from real world to
            simulation coords (for numerical reasons)
        scale_real_to_vis: scale factor to go from real world to
            visualisation coords (meter to pixels)
    """

    def __init__(self, world, position=None, rotation=None):
        super().__init__(world=world, position=position, rotation=rotation, radius=self._radius)

        # 0 .. 255
        self._motor_left = 0
        self._motor_right = 0
        self._light_measurement = 0

        self.body_color = (127, 127, 127, 255)
        self.highlight_color = (255, 0, 0, 255)

    def step(self):
        raise NotImplementedError('Kilobot subclass needs to implement step')

    def set_velocities(self):
        factor_left = self._motor_left / 255.0
        factor_right = self._motor_right / 255.0

        # TODO probably not the right way to move the kilobots
        # TODO look at kilombo
        linear_factor = 0.5 * (factor_left + factor_right)
        angular_factor = factor_right - factor_left

        self.body.linearVelocity = self.body.GetWorldVector(
            (0.0, linear_factor * self.sim_max_lin_vel))
        self.body.angularVelocity = angular_factor * self._max_angular_velocity

    def set_motor(self, left, right):
        self._motor_left = left
        self._motor_right = right

    def get_position(self):
        pos = self.body.position
        return np.array([pos[0], pos[1]]).reshape(1, 2) / self.scale_real_to_sim

    def sense_light(self, light):
        self._light_measurement = light.get_value(self.get_position())
        # TODO implement
        pass

    def draw(self, viewer: rendering.Viewer):
        super(Kilobot, self).draw(viewer)

        top = self.body.GetWorldPoint((0.0, self._radius))
        w = 0.1 * self._radius
        h = np.cos(np.arcsin(w)) - self._radius
        bottom_left = self.body.GetWorldPoint((-w, -h))
        bottom_right = self.body.GetWorldPoint((w, -h))

        viewer.draw_polygon((top, bottom_left, bottom_right), color=(255, 212, 0))


class PhototaxisKilobot(Kilobot):
    def __init__(self, world, pos):
        super(PhototaxisKilobot, self).__init__(position=pos, world=world)

        # self.scale_sim_to_real = 1.0 / scale_real_to_sim

        self.last_light = 0
        self.turn_cw = 1
        self.counter = 0

        # self.env = env

    def step(self):
        # TODO better method to simulate the ambient light sensor
        # (multiple lights?)
        light = self.env['light_pos']

        pos_real = self.scale_sim_to_real * self.body.GetWorldPoint((0.0, self.sim_radius))
        dist = (pos_real - Box2D.b2Vec2(light[0, 0], light[0, 1])).length

        current_light = 1.0 - dist

        # TODO better phototaxis algorithm?
        if dist > 0.01:
            if current_light < self.last_light:
                self.counter = 0
                if self.turn_cw == 1:
                    self.turn_cw = 0
                else:
                    self.turn_cw = 1
            else:
                self.counter = self.counter + 1

            self.last_light = current_light

            if self.counter > 50:
                other = 30  # 150
            else:
                other = 0

            other = 0

            if self.turn_cw == 1:
                self.set_motor(255, other)
            else:
                self.set_motor(other, 255)
        else:
            self.set_motor(0, 0)
