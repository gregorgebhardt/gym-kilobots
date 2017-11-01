import numpy as np

from gym import spaces

from gym.envs.classic_control import rendering


class Light(object):
    def __init__(self):
        self.action_space = None
        
    def step(self, action):
        raise NotImplementedError

    def get_value(self, position: np.ndarray):
        raise NotImplementedError

    def draw(self, viewer: rendering.Viewer):
        raise NotImplementedError


class CircularGradientLight(Light):
    def __init__(self, position: np.ndarray = None, radius=20):
        super(CircularGradientLight, self).__init__()
        self._position = position
        self._radius = radius

        self.action_space = spaces.Box(np.array([-1., -1.]), np.array([1., 1.]))

    def step(self, action: np.ndarray):
        self._position += action

    def get_value(self, position: np.ndarray):
        # TODO implement properly
        distance = np.linalg.norm(self._position - position)
        return 255 * np.minimum(distance / self._radius, 1.)

    def draw(self, viewer: rendering.Viewer):
        t = rendering.Transform(translation=self._position)
        viewer.draw_circle(self._radius, color=(255, 255, 255, 130)).add_attr(t)
