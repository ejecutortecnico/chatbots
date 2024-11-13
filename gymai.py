""" import gymnasium as gym

# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(1000):
    # this is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close() """


# import gymnasium as gym
# import ale_py

# gym.register_envs(ale_py)  # unnecessary but helpful for IDEs

# env = gym.make('ALE/Breakout-v5', render_mode="human")  # remove render_mode in training
# obs, info = env.reset()
# episode_over = False
# while not episode_over:
#     action = env.action_space.sample() #policy(obs)  # to implement - use `env.action_space.sample()` for a random policy
#     obs, reward, terminated, truncated, info = env.step(action)

#     episode_over = terminated or truncated
# env.close()

from ale_py import ALEInterface, roms

ale = ALEInterface()
ale.loadROM(roms.get_rom_path("breakout"))
ale.reset_game()

reward = ale.act(0)  # noop
screen_obs = ale.getScreenRGB()