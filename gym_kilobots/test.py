import gym
import gym_kilobots

import pandas as pd

import numpy as np

# env = gym.make('Kilobots-Empty-Unite-v0')
env = gym.make('Kilobots-GradientLight-v0')
env.reset()

# kilobot_index, object_index, light_index = env.get_index()
# kilobot_frames = []
# object_frames = []
# light_frames = []
#
# reward = []

for t in range(2000):
    env.render()
    if t < 500:
        state, reward, done, info = env.step(np.array([-.25 * np.pi]))
    else:
        state, reward, done, info = env.step(np.array([.75 * np.pi]))
    # state_t, reward_t, done, info = env.step(env.action_space.sample())

    # df_kilobots_t = pd.DataFrame(data=state_t['kilobots'].reshape((1, -1)), columns=kilobot_index, index=[t])
    # df_objects_t = pd.DataFrame(data=state_t['objects'].reshape((1, -1)), columns=object_index, index=[t])
    # df_light_t = pd.DataFrame(data=state_t['light'].reshape((1, -1)), columns=light_index, index=[t])
    #
    # kilobot_frames.append(df_kilobots_t)
    # object_frames.append(df_objects_t)
    # light_frames.append(df_light_t)
    #
    # reward.append(reward_t)

    # if done:
    #     break

# df_kilobots = pd.concat(kilobot_frames)
# df_objects = pd.concat(object_frames)
# df_light = pd.concat(light_frames)
