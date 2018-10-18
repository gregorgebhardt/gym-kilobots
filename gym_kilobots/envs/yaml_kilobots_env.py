import numpy as np

import yaml

from gym import spaces

import gym_kilobots
from gym_kilobots.lib import CircularGradientLight, GradientLight, Quad, CornerQuad, Triangle, Circle, LForm, TForm, \
    CForm, CompositeLight
from .kilobots_env import KilobotsEnv, UnknownLightTypeException, UnknownObjectException


class EnvConfiguration(yaml.YAMLObject):
    yaml_tag = '!EvalEnv'

    class ObjectConfiguration(yaml.YAMLObject):
        yaml_tag = '!ObjectConf'

        def __init__(self, idx, shape, width, height, init):
            self.idx = idx
            self.shape = shape
            self.width = width
            self.height = height
            self.init = init

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

    class KilobotsConfiguration(yaml.YAMLObject):
        yaml_tag = '!KilobotsConf'

        def __init__(self, num, mean, std, type='SimplePhototaxisKilobot'):
            self.num = num
            self.mean = mean
            self.std = std
            self.type = type

    def __init__(self, width, height, resolution, objects, light, kilobots):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.objects = [self.ObjectConfiguration(**obj) for obj in objects]
        self.light = self.LightConfiguration(**light)
        self.kilobots = self.KilobotsConfiguration(**kilobots)


def rot_matrix(alpha):
    return np.array([[np.cos(alpha), -np.sin(alpha)],
                     [np.sin(alpha), np.cos(alpha)]])


class YamlKilobotsEnv(KilobotsEnv):
    def __new__(cls, configuration, *args, **kwargs):
        cls.world_width = configuration.width
        cls.world_height = configuration.height
        cls.world_size = cls.world_width, cls.world_height

        cls.screen_width = int(configuration.resolution * configuration.width)
        cls.screen_height = int(configuration.resolution * configuration.height)
        cls.screen_size = cls.screen_width, cls.screen_width

        return super(YamlKilobotsEnv, cls).__new__(cls, *args, **kwargs)

    def __init__(self, configuration):
        self.conf = configuration
        self.num_kilobots = self.conf.kilobots.num

        super().__init__()

        _state_space_low = self.kilobots_space.low
        _state_space_high = self.kilobots_space.high
        if self.light_state_space:
            _state_space_low = np.concatenate((_state_space_low, self.light_state_space.low))
            _state_space_high = np.concatenate((_state_space_high, self.light_state_space.high))
        if self.object_state_space:
            _state_space_low = np.concatenate((_state_space_low, self.object_state_space.low))
            _state_space_high = np.concatenate((_state_space_high, self.object_state_space.high))

        self.state_space = spaces.Box(low=_state_space_low, high=_state_space_high, dtype=np.float32)

        _observation_spaces_low = self.kilobots_space.low
        _observation_spaces_high = self.kilobots_space.high
        if self.light_observation_space:
            _observation_spaces_low = np.concatenate((_observation_spaces_low, self.light_observation_space.low))
            _observation_spaces_high = np.concatenate((_observation_spaces_high, self.light_observation_space.high))
        if self.object_observation_space:
            # the objects are observed as x, y, sin(theta), cos(theta)
            objects_low = np.array([self.world_x_range[0], self.world_y_range[0], -1., -1.] * len(self._objects))
            objects_high = np.array([self.world_x_range[1], self.world_y_range[1], 1., 1.] * len(self._objects))
            _observation_spaces_low = np.concatenate((_observation_spaces_low, objects_low))
            _observation_spaces_high = np.concatenate((_observation_spaces_high, objects_high))

        self.observation_space = spaces.Box(low=_observation_spaces_low, high=_observation_spaces_high,
                                            dtype=np.float32)


    # def transform_world_to_object_point(self, point, object_idx=0):
    #     return self._objects[object_idx].get_local_point(point)
    #
    # def transform_world_to_object_pose(self, pose, object_idx=0):
    #     return self._objects[object_idx].get_local_pose(pose)
    #
    # def transform_object_to_world_point(self, point, object_idx=0):
    #     return self._objects[object_idx].get_world_point(point)

    def _configure_environment(self):
        for o in self.conf.objects:
            self._init_object(o.shape, o.width, o.height, o.init)

        objects_low = np.array([self.world_x_range[0], self.world_y_range[0], -np.inf] * len(self._objects))
        objects_high = np.array([self.world_x_range[1], self.world_y_range[1], np.inf] * len(self._objects))
        self.object_state_space = spaces.Box(low=objects_low, high=objects_high, dtype=np.float64)
        objects_obs_low = np.array([self.world_x_range[0], self.world_y_range[0], -1., -1.] * len(self._objects))
        objects_obs_high = np.array([self.world_x_range[1], self.world_y_range[1], 1., 1.] * len(self._objects))
        self.object_observation_space = spaces.Box(low=objects_obs_low, high=objects_obs_high, dtype=np.float64)

        self._init_light()

        self.action_space = self._light.action_space

        self._init_kilobots(self.conf.kilobots.num, self.conf.kilobots.mean, self.conf.kilobots.std)

    def _init_object(self, object_shape, object_width, object_height, object_init):
        if object_init == 'random':
            init_position = np.random.rand(2) * np.asarray(self.world_size) + self.world_bounds[0]
            init_position *= 0.8
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

        self._add_object(obj)

    def _init_light(self):
        if self.conf.light.type == 'circular':
            light_bounds = np.array(self.world_bounds) * 1.2
            action_bounds = np.array([-1, -1]) * .01, np.array([1, 1]) * .01

            if self.conf.light.init == 'random':
                init_position = np.random.rand(2) * np.asarray(self.world_size) + self.world_bounds[0]
            else:
                init_position = self.conf.light.init

            self._light = CircularGradientLight(position=init_position, radius=self.conf.light.radius,
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

        self.light_state_space = self._light.observation_space
        if self._observe_light:
            self.light_observation_space = self._light.observation_space

    def _init_kilobots(self, num_kilobots, spawn_mean, spawn_std, type='SimplePhototaxisKilobot'):
        if isinstance(spawn_mean, str) and spawn_mean == 'light':
            if self.conf.light.type == 'circular':
                spawn_mean = self._light.get_state()
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

        # construct state space, observation space and action space
        kb_low = np.array([self.world_x_range[0], self.world_y_range[0]] * len(self._kilobots))
        kb_high = np.array([self.world_x_range[1], self.world_y_range[1]] * len(self._kilobots))
        self.kilobots_space = spaces.Box(low=kb_low, high=kb_high, dtype=np.float64)

    def get_reward(self, state, action, new_state):
        return .0
