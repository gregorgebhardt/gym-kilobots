import numpy as np

import yaml

from gym import spaces

import gym_kilobots
from gym_kilobots.lib import CircularGradientLight, GradientLight, Quad, CornerQuad, Triangle, Circle, LForm, TForm, \
    CForm, CompositeLight
from gym_kilobots.lib.light import MomentumLight, SinglePositionLight
from .kilobots_env import KilobotsEnv, UnknownLightTypeException, UnknownObjectException


class EnvConfiguration(yaml.YAMLObject):
    yaml_tag = '!EvalEnv'

    class ObjectConfiguration(yaml.YAMLObject):
        yaml_tag = '!ObjectConf'

        def __init__(self, idx, color, shape, width, height, init):
            self.idx = idx
            self.shape = shape
            self.width = width
            self.height = height
            self.init = init
            self.color = color

        def __eq__(self, other):
            for k in self.__dict__:
                if k not in other.__dict__:
                    return False
                if not self.__getattribute__(k) == other.__getattribute__(k):
                    return False
            return True

        @property
        def object_type(self):
            _type = self.shape
            if _type in ['corner_quad', 'corner-quad', 'quad']:
                _type = 'square'
            return _type

    class LightConfiguration(yaml.YAMLObject):
        yaml_tag = '!LightConf'

        def __init__(self, obj_type, init, radius=None):
            self.type = obj_type
            self.init = init
            self.radius = radius

        def __eq__(self, other):
            for k in self.__dict__:
                if k not in other.__dict__:
                    return False
                if not self.__getattribute__(k) == other.__getattribute__(k):
                    return False
            return True

    class KilobotsConfiguration(yaml.YAMLObject):
        yaml_tag = '!KilobotsConf'

        def __init__(self, num, mean, std, type='SimplePhototaxisKilobot'):
            self.num = num
            self.mean = mean
            self.std = std
            self.type = type

        def __eq__(self, other):
            for k in self.__dict__:
                if k not in other.__dict__:
                    return False
                if not self.__getattribute__(k) == other.__getattribute__(k):
                    return False
            return True

    def __init__(self, width, height, resolution, objects, light, kilobots):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.objects = [self.ObjectConfiguration(**obj) for obj in objects]
        self.light = self.LightConfiguration(**light)
        self.kilobots = self.KilobotsConfiguration(**kilobots)

    def __eq__(self, other):
        for k in self.__dict__:
            if k not in other.__dict__:
                return False
            if not self.__getattribute__(k) == other.__getattribute__(k):
                return False
        return True


def rot_matrix(alpha):
    return np.array([[np.cos(alpha), -np.sin(alpha)],
                     [np.sin(alpha), np.cos(alpha)]])


class YamlKilobotsEnv(KilobotsEnv):
    def __new__(cls, *, configuration, **kwargs):
        cls.world_width = configuration.width
        cls.world_height = configuration.height
        cls.world_size = cls.world_width, cls.world_height

        cls.screen_width = int(configuration.resolution * configuration.width)
        cls.screen_height = int(configuration.resolution * configuration.height)
        cls.screen_size = cls.screen_width, cls.screen_width

        return super(YamlKilobotsEnv, cls).__new__(cls, **kwargs)

    def __eq__(self, other):
        return self.conf == other.conf

    def __init__(self, *, configuration, **kwargs):
        self.conf = configuration

        super().__init__(**kwargs)

    def _configure_environment(self):
        self._init_objects()
        self._init_light()
        self._init_kilobots()

    @property
    def state_space(self):
        _state_space_low = self.kilobots_state_space.low
        _state_space_high = self.kilobots_state_space.high
        if self.light_state_space:
            _state_space_low = np.concatenate((_state_space_low, self.light_state_space.low))
            _state_space_high = np.concatenate((_state_space_high, self.light_state_space.high))
        if self.object_state_space:
            _state_space_low = np.concatenate((_state_space_low, self.object_state_space.low))
            _state_space_high = np.concatenate((_state_space_high, self.object_state_space.high))

        return spaces.Box(low=_state_space_low, high=_state_space_high, dtype=np.float32)

    @property
    def observation_space(self):
        _observation_spaces_low = self.kilobots_state_space.low
        _observation_spaces_high = self.kilobots_state_space.high
        if self.light_observation_space:
            _observation_spaces_low = np.concatenate((_observation_spaces_low, self.light_observation_space.low))
            _observation_spaces_high = np.concatenate((_observation_spaces_high, self.light_observation_space.high))
        if self.object_observation_space:
            # the objects are observed as x, y, sin(theta), cos(theta)
            objects_low = np.array([self.world_x_range[0], self.world_y_range[0], -1., -1.] * len(self._objects))
            objects_high = np.array([self.world_x_range[1], self.world_y_range[1], 1., 1.] * len(self._objects))
            _observation_spaces_low = np.concatenate((_observation_spaces_low, objects_low))
            _observation_spaces_high = np.concatenate((_observation_spaces_high, objects_high))

        return spaces.Box(low=_observation_spaces_low, high=_observation_spaces_high,
                                            dtype=np.float32)

    def _init_objects(self):
        for o in self.conf.objects:
            self._init_object(o.shape, o.width, o.height, o.init, o.color)

    @property
    def object_state_space(self):
        objects_low = np.array([self.world_x_range[0], self.world_y_range[0], -np.inf] * len(self._objects))
        objects_high = np.array([self.world_x_range[1], self.world_y_range[1], np.inf] * len(self._objects))
        return spaces.Box(low=objects_low, high=objects_high, dtype=np.float64)

    @property
    def object_observation_space(self):
        objects_obs_low = np.array([self.world_x_range[0], self.world_y_range[0], -1., -1.] * len(self._objects))
        objects_obs_high = np.array([self.world_x_range[1], self.world_y_range[1], 1., 1.] * len(self._objects))
        return spaces.Box(low=objects_obs_low, high=objects_obs_high, dtype=np.float64)

    def _init_object(self, object_shape, object_width, object_height, object_init, object_color=None):
        if object_init == 'random':
            init_position = np.random.rand(2) * np.asarray(self.world_size) + self.world_bounds[0]
            init_position *= 0.7
            init_orientation = np.random.rand() * 2 * np.pi - np.pi
            object_init = np.r_[init_position, init_orientation]

        if object_shape in ['square', 'quad', 'rect']:
            obj = Quad(width=object_width, height=object_height,
                       position=object_init[:2], orientation=object_init[2],
                       world=self.world)
        elif object_shape in ['corner_quad', 'corner-quad']:
            obj = CornerQuad(width=object_width, height=object_height,
                             position=object_init[:2], orientation=object_init[2],
                             world=self.world)
        elif object_shape == 'triangle':
            obj = Triangle(width=object_width, height=object_height,
                           position=object_init[:2], orientation=object_init[2],
                           world=self.world)
        elif object_shape == 'circle':
            obj = Circle(radius=object_width, position=object_init[:2],
                         orientation=object_init[2], world=self.world)
        elif object_shape == 'l_shape':
            obj = LForm(width=object_width, height=object_height,
                        position=object_init[:2], orientation=object_init[2],
                        world=self.world)
        elif object_shape == 't_shape':
            obj = TForm(width=object_width, height=object_height,
                        position=object_init[:2], orientation=object_init[2],
                        world=self.world)
        elif object_shape == 'c_shape':
            obj = CForm(width=object_width, height=object_height,
                        position=object_init[:2], orientation=object_init[2],
                        world=self.world)
        else:
            raise UnknownObjectException('Shape of form {} not known.'.format(object_shape))

        if object_color:
            obj.set_color(object_color)
        self._add_object(obj)

    def _init_light(self):
        if not hasattr(self.conf, 'light'):
            return

        if self.conf.light.type in ['circular', 'momentum']:
            light_bounds = np.array(self.world_bounds) * 1.2
            action_bounds = np.array([-1, -1]) * .01, np.array([1, 1]) * .01

            if self.conf.light.init == 'random':
                init_position = np.random.rand(2) * np.asarray(self.world_size) + self.world_bounds[0]
            elif self.conf.light.init == 'object':
                which_object = self._objects[np.random.choice(len(self._objects), 1)[0]]
                init_position = which_object.get_position()
                radius = 1.2 * max(which_object.width, which_object.height) / 2
                angle = np.random.rand() * 2 * np.pi - np.pi
                init_position += (np.cos(angle) * radius, np.sin(angle) * radius)
            else:
                init_position = self.conf.light.init

            if self.conf.light.type == 'circular':
                self._light = CircularGradientLight(position=init_position, radius=self.conf.light.radius,
                                                    bounds=light_bounds, action_bounds=action_bounds)
            elif self.conf.light.type == 'momentum':
                init_angle = np.random.rand() * 2 * np.pi - np.pi
                init_velocity = np.array([np.sin(init_angle), np.cos(init_angle)]) * .01
                max_velocity = .01
                action_bounds = np.array([-1, -1]) * .01, np.array([1, 1]) * .01
                self._light = MomentumLight(position=init_position, velocity=init_velocity,
                                            max_velocity=max_velocity, radius=self.conf.light.radius,
                                            bounds=light_bounds, action_bounds=action_bounds)

            self._light_observation_space = self._light.observation_space
            self._light_state_space = self._light.observation_space

        elif self.conf.light.type == 'linear':
            # sample initial angle from a uniform between -pi and pi
            self._light = GradientLight(angle=self.conf.light.init)

            self._light_state_space = self._light.observation_space
        elif self.conf.light.type == 'composite':
            light_bounds = np.array(self.world_bounds) * 1.2
            action_bounds = np.array([-1, -1]) * .01, np.array([1, 1]) * .01
            lights = []
            for _c in self.conf.light.components:
                if _c['init'] == 'random':
                    _init = np.random.rand(2) * np.asarray(self.world_size) + self.world_bounds[0]
                else:
                    _init = _c['init']
                lights.append(CircularGradientLight(position=_init, radius=_c['radius'], bounds=light_bounds,
                                                    action_bounds=action_bounds))
            self._light = CompositeLight(lights)
        else:
            raise UnknownLightTypeException()

    @property
    def action_space(self):
        if self._light:
            return self._light.action_space
        return None

    @property
    def light_state_space(self):
        if self._light:
            return self._light.observation_space
        return None

    @property
    def light_observation_space(self):
        if self._light and self._observe_light:
            return self._light.observation_space
        return None

    def _init_kilobots(self, type='SimplePhototaxisKilobot'):
        num_kilobots = self.conf.kilobots.num
        spawn_mean = self.conf.kilobots.mean
        spawn_std = self.conf.kilobots.std

        if isinstance(spawn_mean, str) and spawn_mean == 'light':
            if isinstance(self._light, SinglePositionLight):
                spawn_mean = self._light.get_position()
            elif self.conf.light.type == 'composite':
                lights_positions = np.asarray([_l.get_state for _l in self._light.lights])
                idx = np.random.choice(np.arange(len(lights_positions)), num_kilobots)
                spawn_mean = lights_positions[idx]
            else:
                spawn_mean = 'random'
        if isinstance(spawn_mean, str) and spawn_mean == 'random':
            spawn_mean = np.random.rand(2) * np.asarray(self.world_size) + self.world_bounds[0]
            spawn_mean *= 0.9

        # draw the kilobots positions from a normal with mean and variance selected above
        kilobot_positions = np.random.normal(scale=spawn_std, size=(num_kilobots, 2))
        kilobot_positions += spawn_mean

        # assert for each kilobot that it is within the world bounds and add kilobot to the world
        for position in kilobot_positions:
            position = np.maximum(position, self.world_bounds[0] + 0.02)
            position = np.minimum(position, self.world_bounds[1] - 0.02)
            kb_class = getattr(gym_kilobots.lib, type)
            self._add_kilobot(kb_class(self.world, position=position, light=self._light))

    @property
    def kilobots_state_space(self):
        kb_low = np.array([self.world_x_range[0], self.world_y_range[0]] * len(self._kilobots))
        kb_high = np.array([self.world_x_range[1], self.world_y_range[1]] * len(self._kilobots))
        return spaces.Box(low=kb_low, high=kb_high, dtype=np.float64)

    @property
    def kilobots_observation_space(self):
        kb_low = np.array([self.world_x_range[0], self.world_y_range[0]] * len(self._kilobots))
        kb_high = np.array([self.world_x_range[1], self.world_y_range[1]] * len(self._kilobots))
        return spaces.Box(low=kb_low, high=kb_high, dtype=np.float64)

    def get_reward(self, state, action, new_state):
        return .0
