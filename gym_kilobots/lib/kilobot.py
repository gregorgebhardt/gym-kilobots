import numpy as np
from Box2D import b2Vec2

from .body import Circle

from gym.envs.classic_control import rendering

import math, time


class Kilobot(Circle):
    # all parameters in real world units
    _radius = 0.0165  # meters
    _leg_front = np.array([.0, _radius])
    _leg_left = np.array([-0.013, -.009])
    _leg_right = np.array([+0.013, -.009])
    _light_sensor = np.array([.0, -_radius])

    _impulse_right = _leg_front - _leg_right
    _impulse_left = _leg_front - _leg_left

    _max_linear_velocity = 0.01  # meters / s
    _max_angular_velocity = 0.1 * math.pi  # radians / s

    _density = 1.0
    _friction = 0.2
    _restitution = 0.0

    _linear_damping = 0.8
    _angular_damping = 0.8

    def __init__(self, world, position=None, rotation=None, lights=None):
        super().__init__(world=world, position=position, rotation=rotation, radius=self._radius)

        # 0 .. 255
        self._motor_left = 0
        self._motor_right = 0
        self.__light_measurement = 0

        self._body_color = (.5, .5, .5)
        self._highlight_color = (1., 1., 1.)

        self._lights = lights

        self._setup()

    def get_ambientlight(self):
        if self._lights is not None:
            sensor_position = self._body.GetWorldPoint((0.0, -self._radius))
            light_measurements = [l.get_value(sensor_position) for l in self._lights]
            # todo add noise here
            return int(max(light_measurements))
        else:
            return 0

    def set_motors(self, left, right):
        self._motor_left = left
        self._motor_right = right

    def set_color(self, color):
        self._highlight_color = color

    def step(self, time_step):
        # loop kilobot logic
        self._loop()

        cos_dir = np.cos(self.get_orientation())
        sin_dir = np.sin(self.get_orientation())

        linear_velocity = np.zeros(2)
        angular_velocity = .0

        # compute new kilobot position or kilobot velocity
        if self._motor_left and self._motor_right:
            linear_velocity = (self._motor_right + self._motor_left) / 510. * self._max_linear_velocity
            linear_velocity = np.array([sin_dir, cos_dir]) * linear_velocity

            angular_velocity = (self._motor_right - self._motor_left) / 510. * self._max_angular_velocity

        elif self._motor_right:
            angular_velocity = self._motor_right / 255. * self._max_angular_velocity
            next_orientation = self.get_orientation() + angular_velocity * time_step

            linear_velocity = self._radius * np.array([(cos_dir - np.cos(next_orientation)),
                                                       (-sin_dir + np.sin(next_orientation))])

        elif self._motor_left:
            angular_velocity = - self._motor_left / 255. * self._max_angular_velocity
            next_orientation = self.get_orientation() + angular_velocity * time_step

            linear_velocity = self._radius * np.array([(-cos_dir + np.cos(next_orientation)),
                                                       (sin_dir - np.sin(next_orientation))])

        self._body.angularVelocity = angular_velocity
        self._body.linearVelocity = b2Vec2(*linear_velocity.astype(float))

    def draw(self, viewer: rendering.Viewer):
        super(Kilobot, self).draw(viewer)

        top = self._body.GetWorldPoint((0.0, self._radius-.003))
        # w = 0.1 * self._radius
        # h = np.cos(np.arcsin(w)) - self._radius
        bottom_left = self._body.GetWorldPoint((-0.006, -0.009))
        bottom_right = self._body.GetWorldPoint((0.006, -0.009))

        viewer.draw_polygon((top, bottom_left, bottom_right), color=self._highlight_color)

        t = rendering.Transform(translation=self._body.GetWorldPoint((0.0, -self._radius)))
        viewer.draw_circle(.005, res=100, color=(1, 1, 0)).add_attr(t)

    def _setup(self):
        raise NotImplementedError('Kilobot subclass needs to implement _setup')

    def _loop(self):
        raise NotImplementedError('Kilobot subclass needs to implement _loop')


class PhototaxisKilobot(Kilobot):
    def __init__(self, *args, **kwargs):
        super(PhototaxisKilobot, self).__init__(*args, **kwargs)

        self.__light_measurement = 0
        self.__threshold = 1024
        self.__turn_direction = None
        self.__last_update = .0
        self.__update_interval = 1.

    def _setup(self):
        self.__turn_left()

    def _loop(self):
        now = time.monotonic()
        if now - self.__last_update < self.__update_interval:
            return

        print(now)
        self.__last_update = now

        self.__light_measurement = self.get_ambientlight()
        print(self.__light_measurement)

        if self.__light_measurement < self.__threshold:
            self.__threshold = self.__light_measurement

        else:
            self.__threshold = self.__light_measurement
            self.__switch_directions()

    def __switch_directions(self):
        if self.__turn_direction == 'left':
            self.__turn_right()
        else:
            self.__turn_left()

    def __turn_right(self):
        self.__turn_direction = 'right'
        self.set_motors(0, 255)
        self.set_color((255, 0, 0))

    def __turn_left(self):
        self.__turn_direction = 'left'
        self.set_motors(255, 0)
        self.set_color((0, 255, 0))
