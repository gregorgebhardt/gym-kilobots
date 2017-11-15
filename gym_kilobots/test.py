import gym
import gym_kilobots

import numpy as np

env = gym.make('Kilobots-Empty-Unite-v0')
env.reset()

for t in range(2000):
    env.render()
    # state, reward, done, info = env.step((np.array((.0, .0)),))
    state, reward, done, info = env.step(env.action_space.sample())
    # print("{}: {}".format(t, reward))
