import numpy as np
import pandas as pd

from gym import spaces


class Light(object):
    def __init__(self):
        self.observation_space = None
        
    def step(self, action):
        raise NotImplementedError

    def get_value(self, position: np.ndarray):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def draw(self, viewer):
        raise NotImplementedError

    def get_index(self):
        raise NotImplementedError


class SinglePositionLight(Light):
    def __init__(self, position: np.ndarray = None, bounds: (np.ndarray, np.ndarray) = None):
        super().__init__()
        if position is None:
            self._position = np.array((.0, .0))
        else:
            self._position = position

        if bounds is not None:
            self._bounds = bounds
        else:
            self._bounds = np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf])

        self.observation_space = spaces.Box(*self._bounds, dtype=np.float64)

    def step(self, action: np.ndarray):
        self._position += action
        self._position = np.maximum(self._position, self._bounds[0])
        self._position = np.minimum(self._position, self._bounds[1])

    def get_value(self, position: np.ndarray):
        return -np.linalg.norm(self._position - position)

    def get_state(self):
        return self._position

    def get_index(self):
        return pd.Index(['x', 'y'])

    def draw(self, viewer):
        viewer.draw_aacircle(position=self._position, radius=.01, color=(255, 30, 30, 150))


class CircularGradientLight(SinglePositionLight):
    def __init__(self, position: np.ndarray = None, bounds: (np.ndarray, np.ndarray) = None, radius=.2):
        super().__init__(position, bounds)
        self._radius = radius

    def get_value(self, position: np.ndarray):
        distance = np.linalg.norm(self._position - position)
        if distance > self._radius:
            return 0
        return 255 * np.minimum(1 - distance / self._radius, 1.)

    def get_state(self):
        return self._position

    def get_index(self):
        return pd.Index(['x', 'y'])

    def draw(self, viewer):
        viewer.draw_aacircle(position=self._position, radius=self._radius, color=(255, 255, 30, 150))


class SmoothGridLight(Light):
    def __init__(self):
        super(SmoothGridLight, self).__init__()

    def step(self, action):
        raise NotImplementedError

    def get_value(self, position: np.ndarray):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def get_index(self):
        raise NotImplementedError

    def draw(self, viewer):
        raise NotImplementedError


class GradientLight(Light):
    # TODO change to angular representation and add angular displacement as action
    def __init__(self, gradient_start: np.ndarray = None, gradient_end: np.ndarray = None,
                 gradient_min: int = 0, gradient_max: int = 1024):
        super().__init__()

        self._gradient_start = gradient_start
        if self._gradient_start is None:
            self._gradient_start = np.array([0, 0])

        self._gradient_end = gradient_end
        if self._gradient_end is None:
            self._gradient_end = np.array([0, 1])

        self._gradient_vec = self._gradient_end - self._gradient_start
        assert gradient_max > gradient_min
        self._gradient_min = gradient_min
        self._gradient_max = gradient_max
        self._gradient_range = gradient_max - gradient_min

    def step(self, action):
        pass

    def get_value(self, position: np.ndarray):
        query_point = position - self._gradient_start.astype(float)
        projection = self._gradient_vec.dot(query_point)
        projection /= np.linalg.norm(self._gradient_vec)**2

        if projection < 0:
            return self._gradient_min

        return min(projection * self._gradient_range + self._gradient_min, self._gradient_max)

    def get_state(self):
        return None

    def get_index(self):
        raise NotImplementedError

    def draw(self, viewer):
        # viewer.draw_polyline((self._gradient_start, self._gradient_end), color=(1, 0, 0))
        pass
