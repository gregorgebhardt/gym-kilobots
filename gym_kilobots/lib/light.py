import numpy as np

from gym import spaces

from . import kb_rendering


class Light(object):
    def __init__(self):
        self.action_space = None
        
    def step(self, action):
        raise NotImplementedError

    def get_value(self, position: np.ndarray):
        raise NotImplementedError

    def draw(self, viewer: kb_rendering.KilobotsViewer):
        raise NotImplementedError


class CircularGradientLight(Light):
    def __init__(self, position=None, radius=.2):
        super(CircularGradientLight, self).__init__()
        if position is None:
            self._position = np.array((.0, .0))
        else:
            self._position = position
        self._radius = radius

        self.action_space = spaces.Box(np.array([-.01, -.01]), np.array([.01, .01]))

    def step(self, action: np.ndarray):
        self._position += action

    def get_value(self, position: np.ndarray):
        distance = np.linalg.norm(self._position - position)
        if distance > self._radius:
            return 0
        return 255 * np.minimum(1 - distance / self._radius, 1.)

    def draw(self, viewer: kb_rendering.KilobotsViewer):
        viewer.draw_circle(position=self._position, radius=self._radius, color=(255, 255, 30, 150))


class GradientLight(Light):
    def __init__(self, gradient_start: np.ndarray = np.ndarray([0, 0]), gradient_end: np.ndarray = np.ndarray([0, 1]),
                 gradient_min: int = 0, gradient_max: int = 1024):
        super().__init__()

        self._gradient_start = gradient_start.copy()
        self._gradient_end = gradient_end.copy()
        self._gradient_vec = gradient_end - gradient_start
        assert gradient_max > gradient_min
        self._gradient_min = gradient_min
        self._gradient_max = gradient_max
        self._gradient_range = gradient_max - gradient_min

    def step(self, action):
        pass

    def get_value(self, position: np.ndarray):
        query_point = position - self._gradient_start
        projection = self._gradient_vec.dot(query_point)
        projection /= np.linalg.norm(self._gradient_vec)**2

        if projection < 0:
            return self._gradient_min

        return min(projection * self._gradient_range + self._gradient_min, self._gradient_max)

    def draw(self, viewer: kb_rendering.KilobotsViewer):
        # viewer.draw_polyline((self._gradient_start, self._gradient_end), color=(1, 0, 0))
        pass
