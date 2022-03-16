import gym
import sys
import argparse
from functools import partial
from models.bipedal_walker_model import BipedalWalkerNN
from utils.vectorized import ParallelEnv



# CLI args
def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.lower() in ('true', ):
        return True
    elif isinstance(v, str) and v.lower() in ('false', ):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=-1, help='# of cores to use. -1 means use all cores')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--cvt_samples', type=int, default=25000, help='# of samples for computing cvt clusters. Larger value --> higher quality CVT')
    parser.add_argument('--batch_size', type=int, default=100, help='batch evaluations')
    parser.add_argument('--random_init', type=float, default=0.1, help='proportion of niches to be filled before starting')
    parser.add_argument('--random_init_batch', type=int, default=100, help='batch for random initialization')
    parser.add_argument('--dump_period', type=int, default=10000, help='how often to write results (one generation = one batch)')
    parser.add_argument('--cvt_use_cache', type=str2bool, default=True, help='do we cache results of CVT and reuse?')
    parser.add_argument('--iso_sigma', type=float, default=0.01, help='only useful if you use the "iso_dd" variation operator')
    parser.add_argument('--line_sigma', type=float, default=0.2, help='only useful if you use the "iso_dd" variation operator')

    #TODO: remove "parallel" parameter from compute() method. Should be based on --num_workers

    args = parser.parse_args()
    return args


def make_env(env_name='BipedalWalker-v3'):
    assert env_name in ['BipedalWalker-v3'], 'currently only BipedalWalker-v3 is supported'
    env = gym.make(env_name)
    return env


def model_factory(hidden_size=256, init_type='xavier_uniform'):
    model = BipedalWalkerNN(hidden_size=hidden_size, init_type=init_type)
    model.apply(model.init_weights)
    return model


def main():
    args = parse_args()
    print(args)
    # set up factory function to launch parallel environments
    env_fns = [partial(make_env) for _ in range(args.num_workers)]
    envs = ParallelEnv(
        env_fns,
        args.batch_size,
        args.random_init,
        args.seed
    )
    actor = model_factory()
    while True:
        try:
            envs.evaluate_policy([actor])
        except KeyboardInterrupt:
            break
    return 1


if __name__ == '__main__':
    sys.exit(main())