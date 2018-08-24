import numpy as np

from typing import Iterable, Callable, Optional

from gym import spaces


class Light(object):
    relative_actions = True
    interpolate_actions = True

    def __init__(self):
        self.observation_space = None
        self.action_space = None
        
    def step(self, action):
        raise NotImplementedError

    def get_value(self, position: np.ndarray) -> float:
        raise NotImplementedError

    def get_gradient(self, position: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def draw(self, viewer):
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
        if self._action_bounds is None:
            self._action_bounds = np.array([-0.01, -0.01]), np.array([.01, .01])

        self.action_space = spaces.Box(*self._action_bounds, dtype=np.float64)
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

    def get_gradient(self, position: np.ndarray):
        gradient = self._position - position
        return gradient / np.linalg.norm(gradient)

    def get_state(self):
        return self._position

    def draw(self, viewer):
        viewer.draw_aacircle(position=self._position, radius=.01, color=(255, 30, 30, 150))


class CompositeLight(Light):
    def __init__(self, lights: Iterable[Light] = None, reducer: Callable[[np.ndarray, Optional[int]], float] = np.sum):
        """

        :type lights: Iterable[Light] a list of Light objects
        :type reducer: Callable[[np.ndarray, Optional[int]], float]
        """
        super().__init__()
        self._lights = lights
        self._reducer = reducer

        self.observation_space = spaces.Tuple(l.observation_space for l in self._lights)
        self.action_space = spaces.Box(np.concatenate(list(l.action_space.low for l in self._lights)),
                                       np.concatenate(list(l.action_space.high for l in self._lights)),
                                       dtype=np.float64)
        self._action_dims = list(l.action_space.shape[0] for l in self._lights)

    def step(self, action):
        if action is not None:
            for l, ad in zip(self._lights, self._action_dims):
                l.step(action[:ad])
                action = action[ad:]

    def get_value(self, position: np.ndarray) -> float:
        return self._reducer(np.array([l.get_value(position) for l in self._lights]))

    def get_gradient(self, position: np.ndarray):
        max_l = np.argmax([l.get_value(position) for l in self._lights])
        return self._lights[max_l].get_gradient(position)
        # return self._reducer(np.array([l.get_gradient(position) for l in self._lights]), 0)

    def get_state(self):
        return np.concatenate(list(l.get_state() for l in self._lights))

    def draw(self, viewer):
        for l in self._lights:
            l.draw(viewer)


class CircularGradientLight(SinglePositionLight):
    def __init__(self, radius=.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._radius = radius

    def get_value(self, position: np.ndarray):
        distance = np.linalg.norm(self._position - position)
        if distance > self._radius:
            return 0
        return 255 * np.minimum(1 - distance / self._radius, 1.)

    def get_gradient(self, position: np.ndarray):
        gradient = self._position - position
        norm_gradient = np.linalg.norm(gradient)
        if norm_gradient < self._radius:
            gradient /= norm_gradient
        else:
            gradient = np.zeros(2)
        return gradient

    def get_state(self):
        return self._position

    def draw(self, viewer):
        viewer.draw_aacircle(position=self._position, radius=self._radius, color=(255, 255, 30, 150))


class SmoothGridLight(Light):
    def __init__(self):
        super(SmoothGridLight, self).__init__()

    def step(self, action):
        raise NotImplementedError

    def get_value(self, position: np.ndarray):
        raise NotImplementedError

    def get_gradient(self, position: np.ndarray):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def draw(self, viewer):
        raise NotImplementedError


class GradientLight(Light):
    relative_actions = False

    def __init__(self, angle: float = .0):
        super().__init__()

        self._gradient_angle = np.array([angle])
        self._gradient_vec = np.r_[np.cos(angle), np.sin(angle)]

        self._bounds = np.array([-np.pi]), np.array([np.pi])
        self._action_bounds = 2 * np.array([-np.pi]), 2 * np.array([np.pi])

        self.observation_space = spaces.Box(*self._bounds, dtype=np.float64)
        self.action_space = spaces.Box(*self._action_bounds, dtype=np.float64)

    def step(self, action):
        if action is None:
            return

        action = np.maximum(action, self._action_bounds[0])
        action = np.minimum(action, self._action_bounds[1])
        self._gradient_angle = action
        if self._gradient_angle < self._bounds[0]:
            self._gradient_angle += 2 * np.pi
        if self._gradient_angle > self._bounds[1]:
            self._gradient_angle -= 2 * np.pi

        self._gradient_vec = np.r_[np.cos(self._gradient_angle), np.sin(self._gradient_angle)]

    def get_value(self, position: np.ndarray):
        projection = self._gradient_vec.dot(position)
        return projection

    def get_gradient(self, position: np.ndarray):
        return self._gradient_vec

    def get_state(self):
        return self._gradient_angle

    def draw(self, viewer):
        viewer.draw_polyline((np.array([0, 0]), self._gradient_vec), color=(1, 0, 0))
        pass
