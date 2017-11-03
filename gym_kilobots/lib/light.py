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
    def __init__(self, position: np.ndarray = None, radius=.2):
        super(CircularGradientLight, self).__init__()
        self._position = position
        self._radius = radius

        self.action_space = spaces.Box(np.array([-.01, -.01]), np.array([.01, .01]))

    def step(self, action: np.ndarray):
        self._position += action

    def get_value(self, position: np.ndarray):
        # TODO implement properly
        distance = np.linalg.norm(self._position - position)
        return 255 * np.minimum(distance / self._radius, 1.)

    def draw(self, viewer: rendering.Viewer):
        t = rendering.Transform(translation=self._position)
        viewer.draw_circle(self._radius, color=np.array((255, 255, 30))/255).add_attr(t)


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

    def draw(self, viewer: rendering.Viewer):
        # viewer.draw_polyline((self._gradient_start, self._gradient_end), color=(1, 0, 0))
        pass
