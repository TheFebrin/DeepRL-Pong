import gym

env = gym.make("Pong-v0")
observation = env.reset()
done = False
score = 0
while not done:
    # env.render()
    action = env.action_space.sample(
    )  # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    score += reward

    if done:
        observation = env.reset()
env.close()
print(f'Score: {score}')
