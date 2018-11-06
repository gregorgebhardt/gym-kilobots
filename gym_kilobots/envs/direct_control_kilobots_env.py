import abc

import numpy as np
from gym import spaces
from .kilobots_env import KilobotsEnv


class DirectControlKilobotsEnv(KilobotsEnv):
    def __init__(self, **kwargs):
        super(DirectControlKilobotsEnv, self).__init__(**kwargs)

    @property
    def action_space(self):
        as_low = np.array([kb.action_space.low for kb in self._kilobots])
        as_high = np.array([kb.action_space.high for kb in self._kilobots])
        return spaces.Box(as_low, as_high, dtype=np.float64)

    def step(self, actions: np.ndarray):
        if actions is not None:
            # assert self.action_space.contains(actions), 'actions not in action_space'

            for kb, a in zip(self.kilobots, actions):
                kb.set_action(a)

        else:
            for kb in self.kilobots:
                kb.set_action(None)

        return super(DirectControlKilobotsEnv, self).step(None)
