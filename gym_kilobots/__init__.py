from gym.envs.registration import register

register(
    id='Kilobots-v0',
    entry_point='gym_kilobots.envs:QuadAssemblyKilobotsEnv'
)