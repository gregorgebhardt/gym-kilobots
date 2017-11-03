import gym
import gym_kilobots

env = gym.make('Kilobots-v0')
env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())