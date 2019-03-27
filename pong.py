import gym

env = gym.make('Pong-v0')


env.reset()
observation = env.reset()


for _ in range(100):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

env.reset()
env.close()