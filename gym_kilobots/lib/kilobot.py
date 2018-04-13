import math

import numpy as np
from Box2D import b2Vec2

from .body import Circle


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
    _max_angular_velocity = 0.1 * math.pi  # radians / s

    _density = 1.0
    _friction = 0.0
    _restitution = 0.0

    _linear_damping = .8
    _angular_damping = .8

    def __init__(self, world, position=None, orientation=None, light=None):
        # all parameters in real world units
        super().__init__(world=world, position=position, orientation=orientation, radius=self._radius)

        # 0 .. 255
        self._motor_left = 0
        self._motor_right = 0
        self.__light_measurement = 0
        self.__turn_direction = None

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

    def switch_directions(self):
        if self.__turn_direction == 'left':
            self.turn_right()
        else:
            self.turn_left()

    def turn_right(self):
        self.__turn_direction = 'right'
        self.set_motors(0, 255)
        self.set_color((255, 0, 0))

    def turn_left(self):
        self.__turn_direction = 'left'
        self.set_motors(255, 0)
        self.set_color((0, 255, 0))

    def set_color(self, color):
        self._highlight_color = color

    def step(self, time_step):
        # loop kilobot logic
        self._loop()

        cos_dir = np.cos(self.get_orientation())
        sin_dir = np.sin(self.get_orientation())

        linear_velocity = [.0, .0]
        angular_velocity = .0

        # compute new kilobot position or kilobot velocity
        if self._motor_left and self._motor_right:
            linear_velocity = (self._motor_right + self._motor_left) / 510. * self._max_linear_velocity
            linear_velocity = [sin_dir * linear_velocity, cos_dir * linear_velocity]

            angular_velocity = (self._motor_right - self._motor_left) / 510. * self._max_angular_velocity

        elif self._motor_right:
            angular_velocity = self._motor_right / 255. * self._max_angular_velocity
            angular_displacement = angular_velocity * time_step

            c, s = np.cos(angular_displacement), np.sin(angular_displacement)
            R = [[c, -s], [s, c]]

            translation = self._leg_left - np.dot(R, self._leg_left)
            linear_velocity = self._body.GetWorldVector(translation) / time_step

        elif self._motor_left:
            angular_velocity = -self._motor_left / 255. * self._max_angular_velocity
            angular_displacement = angular_velocity * time_step

            c, s = np.cos(angular_displacement), np.sin(angular_displacement)
            R = [[c, -s], [s, c]]

            translation = self._leg_right - np.dot(R, self._leg_right)
            linear_velocity = self._body.GetWorldVector(translation) / time_step

        self._body.angularVelocity = angular_velocity
        if type(linear_velocity) == np.ndarray:
            self._body.linearVelocity = b2Vec2(*linear_velocity.astype(float))
        else:
            self._body.linearVelocity = linear_velocity

    def draw(self, viewer):
        super(Kilobot, self).draw(viewer)
        # viewer.draw_circle(position=self._body.position, radius=self._radius, color=(50,) * 3, filled=False)

        # draw direction as triangle with color set by function
        top = self._body.GetWorldPoint((0.0, self._radius - .005))
        # w = 0.1 * self._radius
        # h = np.cos(np.arcsin(w)) - self._radius
        # bottom_left = self._body.GetWorldPoint((-0.006, -0.009))
        # bottom_right = self._body.GetWorldPoint((0.006, -0.009))
        middle = self._body.position

        # viewer.draw_polygon(vertices=(top, bottom_left, bottom_right), color=self._highlight_color)
        viewer.draw_polyline(vertices=(top, middle), color=self._highlight_color, closed=False, width=.005)

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

    @classmethod
    def get_radius(cls):
        return cls._radius

    def _setup(self):
        raise NotImplementedError('Kilobot subclass needs to implement _setup')

    def _loop(self):
        raise NotImplementedError('Kilobot subclass needs to implement _loop')


class SimplePhototaxisKilobot(Kilobot):
    def __init__(self, world, position=None, orientation=None, light=None):
        super().__init__(world=world, position=position, orientation=orientation, light=light)

        self.last_light = 0
        self.turn_cw = 1
        self.counter = 0

        self.env = world

    def _setup(self):
        self.turn_left()

    def _loop(self):
        pos_real = np.array(self._body.GetWorldPoint((0.0, -self._radius)))

        dist = np.linalg.norm(pos_real - self._light.get_state())

        current_light = 1.0 - dist

        # TODO better phototaxis algorithm?
        if dist > 0.01:
            if current_light > self.last_light:
                self.counter = 0
                self.switch_directions()
            else:
                self.counter = self.counter + 1

            self.last_light = current_light

        # else:
            # self._set_motors(0, 0)

    def step(self, time_step):
        movement_direction = self._light.get_gradient(self._body.position)

        n = np.sqrt(np.dot(movement_direction, movement_direction))
        # n = np.linalg.norm(movement_direction)
        if n > self._max_linear_velocity:
            movement_direction = movement_direction / n * self._max_linear_velocity

        self._body.linearVelocity = b2Vec2(*movement_direction.astype(float))
        # self._body.angle = np.arctan2(movement_direction[1], movement_direction[0])
        self._body.linearDamping = .0


class PhototaxisKilobot(Kilobot):
    def __init__(self, world, position=None, orientation=None, light=None):
        super(PhototaxisKilobot, self).__init__(world=world, position=position, orientation=orientation, light=light)

        self.__light_measurement = 0
        self.__threshold = -np.inf
        self.__last_update = .0
        self.__update_interval = 6
        self.__update_counter = 0
        self.__no_change_counter = 0
        self.__no_change_threshold = 15

    def _setup(self):
        self.turn_left()

    def _loop(self):

        if self.__update_counter % self.__update_interval:
            self.__update_counter += 1
            return

        self.__update_counter += 1

        self.__light_measurement = self.get_ambientlight()

        if self.__light_measurement > self.__threshold or self.__no_change_counter >= self.__no_change_threshold:
            self.__threshold = self.__light_measurement + .01
            self.switch_directions()
            self.__no_change_counter = 0
        else:
            self.__no_change_counter += 1
