from gym.envs.registration import register

register(
    id='Kilobots-Empty-Unite-v0',
    entry_point='gym_kilobots.envs:UniteKilobotsEnv'
)

register(
    id='Kilobots-QuadAssembly-v0',
    entry_point='gym_kilobots.envs:QuadAssemblyKilobotsEnv'
)
