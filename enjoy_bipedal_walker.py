import sys
import gym
from wrappers.BDWrapper import BDWrapper


def make_env(env_name='BipedalWalker-v3'):
    assert env_name in ['BipedalWalker-v3'], 'currently only BipedalWalker-v3 is supported'
    env = gym.make(env_name)
    env = BDWrapper(env)
    return env


def enjoy():
    env = make_env()
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
    return 1


if __name__ == '__main__':
    sys.exit(enjoy())