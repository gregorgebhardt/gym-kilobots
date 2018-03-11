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
    def __init__(self, position: np.ndarray = None, bounds: (np.ndarray, np.ndarray) = None,
                 action_bounds: (np.ndarray, np.ndarray) = None):
        super().__init__()
        if position is None:
            self._position = np.array((.0, .0))
        else:
            self._position = position

        self._bounds = bounds
        if self._bounds is None:
            self._bounds = np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf])

        self._action_bounds = action_bounds

        self.observation_space = spaces.Box(*self._bounds, dtype=np.float64)

    def step(self, action: np.ndarray):
        if action is None:
            return
        if self._action_bounds:
            action = np.maximum(action, self._action_bounds[0])
            action = np.minimum(action, self._action_bounds[1])
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
    def __init__(self, radius=.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
    def __init__(self, center: np.ndarray = None, angle: float = .0):
        super().__init__()

        self._gradient_center = center
        if self._gradient_center is None:
            self._gradient_center = np.array([0, 0])

        self._gradient_angle = angle
        self._gradient_vec = np.r_[np.cos(angle), np.sin(angle)]

        self._bounds = np.array([-np.pi]), np.array([np.pi])

        self.observation_space = spaces.Box(*self._bounds, dtype=np.float64)
        self.action_space = spaces.Box(*self._bounds, dtype=np.float64)

    def step(self, action):
        if action is None:
            return
        self._gradient_angle = action
        # self._gradient_angle = np.maximum(self._gradient_angle, self._bounds[0])
        # self._gradient_angle = np.minimum(self._gradient_angle, self._bounds[1])
        self._gradient_vec = np.r_[np.cos(action), np.sin(action)]

    def get_value(self, position: np.ndarray):
        query_point = position - self._gradient_center.astype(float)
        projection = self._gradient_vec.dot(query_point)
        return projection

    def get_state(self):
        return self._gradient_vec

    def get_index(self):
        return pd.Index(['theta'])

    def draw(self, viewer):
        viewer.draw_polyline((self._gradient_center, self._gradient_center + self._gradient_vec), color=(1, 0, 0))
        pass
