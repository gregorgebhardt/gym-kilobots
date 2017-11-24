import numpy as np
from Box2D import b2Vec2
from numpy.ma.core import filled

from .body import Circle

from . import kb_rendering

import math, time


class Kilobot(Circle):
    _radius = 0.0165

    _leg_front = np.array([.0, _radius])
    _leg_left = np.array([-0.013, -.009])
    _leg_right = np.array([+0.013, -.009])
    _light_sensor = np.array([.0, -_radius+.001])
    _led = np.array([.011, .01])

    # _impulse_right_dir = _leg_front - _leg_right
    # _impulse_left_dir = _leg_front - _leg_left
    # _impulse_right_point_body = (_leg_front + _leg_right) / 2
    # _impulse_left_point_body = (_leg_front + _leg_left) / 2

    _max_linear_velocity = 0.01  # meters / s
    _max_angular_velocity = 0.2 * math.pi  # radians / s

    _density = 1.0
    _friction = 0.2
    _restitution = 0.0

    _linear_damping = .8
    _angular_damping = .8

    def __init__(self, world, position=None, rotation=None, light=None):
        # all parameters in real world units
        super().__init__(world=world, position=position, rotation=rotation, radius=self._radius)

        # 0 .. 255
        self._motor_left = 0
        self._motor_right = 0
        self.__light_measurement = 0

        self._body_color = (100, 100, 100)
        self._highlight_color = (255, 255, 255)

        self._light = light

        self._setup()

    def get_ambientlight(self):
        if self._light is not None:
            sensor_position = self._body.GetWorldPoint((0.0, -self._radius))
            light_measurement = self._light.get_value(sensor_position)
            # todo add noise here
            return light_measurement
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
            angular_displacement = angular_velocity * time_step

            c, s = np.cos(angular_displacement), np.sin(angular_displacement)
            R = np.array([[c, -s], [s, c]])

            translation = self._leg_left - R.dot(self._leg_left)
            linear_velocity = self._body.GetWorldVector(translation) / time_step

        elif self._motor_left:
            angular_velocity = -self._motor_left / 255. * self._max_angular_velocity
            angular_displacement = angular_velocity * time_step

            c, s = np.cos(angular_displacement), np.sin(angular_displacement)
            R = np.array([[c, -s], [s, c]])

            translation = self._leg_right - R.dot(self._leg_right)
            linear_velocity = self._body.GetWorldVector(translation) / time_step

        self._body.angularVelocity = angular_velocity
        if type(linear_velocity) == np.ndarray:
            self._body.linearVelocity = b2Vec2(*linear_velocity.astype(float))
        else:
            self._body.linearVelocity = linear_velocity

    def draw(self, viewer: kb_rendering.KilobotsViewer):
        super(Kilobot, self).draw(viewer)

        # draw direction as triangle with color set by function
        top = self._body.GetWorldPoint((0.0, self._radius-.003))
        # w = 0.1 * self._radius
        # h = np.cos(np.arcsin(w)) - self._radius
        bottom_left = self._body.GetWorldPoint((-0.006, -0.009))
        bottom_right = self._body.GetWorldPoint((0.006, -0.009))

        viewer.draw_polygon(vertices=(top, bottom_left, bottom_right), color=self._highlight_color)

        # t = rendering.Transform(translation=self._body.GetWorldPoint(self._led))
        # viewer.draw_circle(.003, res=20, color=self._highlight_color).add_attr(t)
        # viewer.draw_circle(.003, res=20, color=(0, 0, 0), filled=False).add_attr(t)

        # light sensor
        # viewer.draw_circle(position=self._body.GetWorldPoint(self._light_sensor), radius=.005, color=(0, 0, 0))
        # viewer.draw_circle(position=self._body.GetWorldPoint(self._light_sensor), radius=.0035, color=(255, 255, 0))

        # draw legs
        # viewer.draw_circle(position=self._body.GetWorldPoint(self._leg_front), radius=.001, color=(0, 0, 0))
        # viewer.draw_circle(position=self._body.GetWorldPoint(self._leg_left), radius=.001, color=(0, 0, 0))
        # viewer.draw_circle(position=self._body.GetWorldPoint(self._leg_right), radius=.001, color=(0, 0, 0))

    def _setup(self):
        raise NotImplementedError('Kilobot subclass needs to implement _setup')

    def _loop(self):
        raise NotImplementedError('Kilobot subclass needs to implement _loop')


class PhototaxisKilobot(Kilobot):
    def __init__(self, *args, **kwargs):
        super(PhototaxisKilobot, self).__init__(*args, **kwargs)

        self.__light_measurement = 0
        self.__threshold = 0
        self.__turn_direction = None
        self.__last_update = .0
        self.__update_interval = 30
        self.__update_counter = 0

    def _setup(self):
        self.__turn_left()

    def _loop(self):

        if self.__update_counter % self.__update_interval:
            self.__update_counter += 1
            return

        self.__update_counter += 1

        self.__light_measurement = self.get_ambientlight()

        if self.__light_measurement > self.__threshold:
            self.__threshold = self.__light_measurement + 1
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
