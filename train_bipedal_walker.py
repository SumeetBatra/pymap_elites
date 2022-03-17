import os.path

import gym
import sys
import argparse
from functools import partial

import torch
import numpy as np

from models.bipedal_walker_model import BipedalWalkerNN
from utils.vectorized import ParallelEnv
from utils.logger import log
from wrappers.BDWrapper import BDWrapper
from map_elites.variation_operators import VariationOperator



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
    parser.add_argument('--max_evals', type=int, default=1e6, help='Total number of evaluations to perform')
    parser.add_argument('--save_path', default='./', type=str, help='path where to save results')
    parser.add_argument('--dim_map', default=2, type=int, help='Dimensionality of the behavior space. Default is 2 for bipedal walker (obviously)')
    parser.add_argument('--save_period', default=10000, type=int, help='How many evaluations b/w saving archives')

    # args for cross over and mutation of agent params
    parser.add_argument('--mutation_op', default=None, type=str, choices=['polynomial_mutation', 'gaussian_mutation', 'uniform_mutation'], help='Type of mutation to perform. Leave as None to do no mutations')
    parser.add_argument('--crossover_op', default='iso_dd', type=str, choices=['sbx', 'iso_dd'], help='Type of crossover operation to perform')
    parser.add_argument("--min_genotype", default=False, type=float, help='Minimum value a gene in the genotype can take (if False no limit) (Set to False in GECCO paper)')
    parser.add_argument("--max_genotype", default=False, type=float, help='Maximum value a gene in the genotype can take (if False no limit) (Set to False in GECCO paper)')
    parser.add_argument('--mutation_rate', default=0.05, type=float, help='probability of gene to be mutated')
    parser.add_argument('--crossover_rate', default=0.75, type=float, help='probability of genotypes being crossed over')
    parser.add_argument("--eta_m", default=5.0, type=float, help='Parameter for polynomaial mutation (Not used in GECCO paper)')
    parser.add_argument("--eta_c", default=10.0, type=float, help='Parameter for Simulated Binary Crossover (Not used in GECCO paper)')
    parser.add_argument("--sigma", default=0.2, type=float, help='Sandard deviation for gaussian muatation (Not used in GECCO paper)')
    parser.add_argument("--iso_sigma", default=0.01, type=float, help='Gaussian parameter in iso_dd/directional variation (sigma_1)')
    parser.add_argument("--line_sigma", default=0.2, type=float, help='Line parameter in iso_dd/directional variation (sigma_2)')
    parser.add_argument("--max_uniform", default=0.1, type=float, help='Max mutation for uniform muatation (Not used in GECCO paper)')
    parser.add_argument('--eval_batch_size', default=100, type=int, help='Batch size for parallel evaluation of policies')
    parser.add_argument('--proportion_evo', default=0.5, type=float, help='Proportion of batch to use in GA variation (crossovers/mutations)')


    #TODO: remove "parallel" parameter from compute() method. Should be based on --num_workers

    args = parser.parse_args()
    return args


def make_env(env_name='BipedalWalker-v3'):
    assert env_name in ['BipedalWalker-v3'], 'currently only BipedalWalker-v3 is supported'
    env = gym.make(env_name)
    env = BDWrapper(env)
    return env


def model_factory(hidden_size=256, init_type='xavier_uniform'):
    model = BipedalWalkerNN(hidden_size=hidden_size, init_type=init_type)
    model.apply(model.init_weights)
    return model


def main():
    args = parse_args()
    cfg = vars(args)
    # make folders
    if not os.path.exists(cfg['save_path']):
        os.mkdir(cfg['save_path'])

    log.debug(f'############## PARAMETERS #########################')
    for key, val in cfg.items():
        log.debug(f'{key}: {val}')
    log.debug('#' * 50)

    filename = f'CVT-MAP-ELITES_BipedalWalkerV3_seed_{cfg["seed"]}_dim_map_{cfg["dim_map"]}'
    file_save_path = os.path.join(cfg['save_path'], filename)
    actors_file = open(file_save_path, 'w')

    # set seeds
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    # set up factory function to launch parallel environments
    env_fns = [partial(make_env) for _ in range(cfg['num_workers'])]
    envs = ParallelEnv(
        env_fns,
        cfg['batch_size'],
        cfg['random_init'],
        cfg['seed']
    )
    # initialize the variation operator (that performs crossovers/mutations)
    variation_op = VariationOperator(num_cpu=cfg['num_workers'],
                                     crossover_op=cfg['crossover_op'],
                                     mutation_op=cfg['mutation_op'],
                                     max_gene=cfg['max_genotype'],
                                     min_gene=cfg['min_genotype'],
                                     mutation_rate=cfg['mutation_rate'],
                                     crossover_rate=cfg['crossover_rate'],
                                     eta_m=cfg['eta_m'],
                                     eta_c=cfg['eta_c'],
                                     sigma=cfg['sigma'],
                                     max_uniform=cfg['max_uniform'],
                                     iso_sigma=cfg['iso_sigma'],
                                     line_sigma=cfg['line_sigma'])
    actor = model_factory()
    while True:
        try:
            envs.evaluate_policy([actor])
        except KeyboardInterrupt:
            break
    return 1


if __name__ == '__main__':
    sys.exit(main())