import sys
import gym


def enjoy():
    env = gym.make("BipedalWalker-v3")
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action = env.action_space.sample()
        obs, rew, done, _ = env.step(action)
    return 1


if __name__ == '__main__':
    sys.exit(enjoy())