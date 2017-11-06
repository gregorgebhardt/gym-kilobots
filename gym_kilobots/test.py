import gym
import gym_kilobots

import numpy as np

env = gym.make('Kilobots-v0')
env.reset()

for _ in range(2000):
    env.render()
    env.step((np.array((.0, .0)),))
    # env.step(env.action_space.sample())
