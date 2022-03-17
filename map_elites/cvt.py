#! /usr/bin/env python
#| This file is a part of the pymap_elites framework.
#| Copyright 2019, INRIA
#| Main contributor(s):
#| Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
#| Eloise Dalin , eloise.dalin@inria.fr
#| Pierre Desreumaux , pierre.desreumaux@inria.fr
#|
#|
#| **Main paper**: Mouret JB, Clune J. Illuminating search spaces by
#| mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.
#|
#| This software is governed by the CeCILL license under French law
#| and abiding by the rules of distribution of free software.  You
#| can use, modify and/ or redistribute the software under the terms
#| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
#| following URL "http://www.cecill.info".
#|
#| As a counterpart to the access to the source code and rights to
#| copy, modify and redistribute granted by the license, users are
#| provided only with a limited warranty and the software's author,
#| the holder of the economic rights, and the successive licensors
#| have only limited liability.
#|
#| In this respect, the user's attention is drawn to the risks
#| associated with loading, using, modifying and/or developing or
#| reproducing the software by the user in light of its specific
#| status of free software, that may mean that it is complicated to
#| manipulate, and that also therefore means that it is reserved for
#| developers and experienced professionals having in-depth computer
#| knowledge. Users are therefore encouraged to load and test the
#| software's suitability as regards their requirements in conditions
#| enabling the security of their systems and/or data to be ensured
#| and, more generally, to use and operate it in the same conditions
#| as regards security.
#|
#| The fact that you are presently reading this means that you have
#| had knowledge of the CeCILL license and that you accept its terms.

import math
import numpy as np
import multiprocessing

# from scipy.spatial import cKDTree : TODO -- faster?
import wandb
import time
import torch
from sklearn.neighbors import KDTree
from models.bipedal_walker_model import BipedalWalkerNN
from itertools import count

from map_elites import common as cm
from utils.logger import log, config_wandb


def __add_to_archive(s, centroid, archive, kdt):
    niche_index = kdt.query([centroid], k=1)[1][0][0]
    niche = kdt.data[niche_index]
    n = cm.make_hashable(niche)
    s.centroid = n
    if n in archive:
        if s.fitness > archive[n].fitness:
            archive[n] = s
            return 1
        return 0
    else:
        archive[n] = s
        return 1


# evaluate a single vector (x) with a function f and return a species
# t = vector, function
def __evaluate(t):
    z, f = t  # evaluate z with function f
    fit, desc = f(z)
    return cm.Species(z, desc, fit)


# map-elites algorithm (CVT variant)
def compute_nn(dim_map, dim_x, f, envs, cfg, actors_file, filename, save_path,
            n_niches=1000,
            max_evals=1e5,
            log_file=None,
            variation_operator=cm.variation):
    """CVT MAP-Elites
       Vassiliades V, Chatzilygeroudis K, Mouret JB. Using centroidal voronoi tessellations to scale up the multidimensional archive of phenotypic elites algorithm. IEEE Transactions on Evolutionary Computation. 2017 Aug 3;22(4):623-30.

       Format of the logfile: evals archive_size max mean median 5%_percentile, 95%_percentile
        dim_map: dimensionality of the map. Ex. % time contact w/ ground of 4 legs = 4 dims, etc
        dim_x: dimensionality of the behavior descriptor
        f: the environment which returns a fitness and behavior descriptor
    """
    # log hyperparams to wandb
    config_wandb(batch_size=cfg['batch_size'], max_evals=max_evals)

    # setup the parallel processing pool
    # TODO: This is probably not needed anymore, since parallelization comes from the ParallelEnv class
    num_cores = cfg['num_workers']
    pool = torch.multiprocessing.Pool(num_cores)

    # create the CVT
    cluster_centers = cm.cvt(n_niches, dim_map,
                             cfg['cvt_samples'], cfg['cvt_use_cache'])
    species_centers = cm.cvt(cfg['n_species'])
    kdt = KDTree(cluster_centers, leaf_size=30, metric='euclidean')
    cm.__write_centroids(cluster_centers)

    archive = {}  # init archive (empty)
    n_evals = 0  # number of evaluations since the beginning
    b_evals = 0  # number evaluation since the last dump

    # main loop
    while (n_evals < max_evals):
        start_time = time.time()
        to_evaluate = []
        # random initialization
        if len(archive) <= cfg['random_init'] * n_niches:  # initialize a |random_init| [0,1] percentage of actors
            log.debug("Initializing the neural network actors' weights from scratch")
            actor = BipedalWalkerNN(hidden_size=128)
            to_evaluate += [actor]
        else:  # variation/selection loop
            log.debug("Selection/Variation loop of existing actors")
            keys = list(archive.keys())
            # we select all the parents at the same time because randint is slow
            rand1 = np.random.randint(len(keys), size=cfg['batch_size'])
            rand2 = np.random.randint(len(keys), size=cfg['batch_size'])
            for n in range(0, cfg['batch_size']):
                # parent selection
                parent1 = archive[keys][rand1[n]]
                parent2 = archive[keys][rand2[n]]
                # copy and add variation
                to_evaluate += variation_operator(archive, cfg['eval_batch_size'], cfg['proportion_evo'])

        # evaluations of the fitness and BD of new batch
        solutions = envs.eval_policy(to_evaluate)
        n_evals += len(to_evaluate)
        b_evals += len(to_evaluate)
        fps = len(to_evaluate) / (time.time() - start_time)
        fps = round(fps, 1)
        log.debug(f'{n_evals/int(cfg["max_evals"])}')
        # add to archive
        for idx, solution in enumerate(solutions):
            # TODO: need to check for if robot is alive?
            agent = Individual(genotype=to_evaluate[idx], phenotype=solution[2], fitness=solution[0])
            added = __add_to_archive(agent, agent.phenotype, archive, kdt)
            if added:
                actors_file.write("{} {} {} {} {} {} {} {} {} {}\n".format(n_evals,
                                                                       agent.genotype.id,
                                                                       agent.fitness,
                                                                       str(agent.phenotype).strip("[]"),
                                                                       str(agent.centroid).strip("()"),
                                                                       agent.genotype.parent_1_id,
                                                                       agent.genotype.parent_2_id,
                                                                       agent.genotype.type,
                                                                       agent.genotype.novel,
                                                                       agent.genotype.delta_f))
                actors_file.flush()

            # save the state of the archive
        if b_evals >= cfg['save_period'] and cfg['save_period'] != -1:
            cm.save_archive(archive, n_evals, filename, save_path)
            b_evals = 0

        # logging
        fit_list = np.array([x.fitness for x in archive.values()])
        # write log
        log.info(f'n_evals: {n_evals}, mean fitness: {np.mean(fit_list)}, median fitness: {np.median(fit_list)}, \
            5th percentile: {np.percentile(fit_list, 5)}, 95th percentile: {np.percentile(fit_list, 95)}')
        log.debug(f'FPS: {fps}')
        wandb.log({
            "evals": n_evals,
            "mean fitness": np.mean(fit_list),
            "median fitness": np.median(fit_list),
            "5th percentile": np.percentile(fit_list, 5),
            "95th percentile": np.percentile(fit_list, 95),
            "fps": fps
        })
        cm.save_archive(archive, n_evals, filename, save_path, save_models=True)
        envs.close()

# map-elites algorithm (CVT variant)
def compute(dim_map, dim_x, f,
            n_niches=1000,
            max_evals=1e5,
            params=cm.default_params,
            log_file=None,
            variation_operator=cm.variation):
    """CVT MAP-Elites
       Vassiliades V, Chatzilygeroudis K, Mouret JB. Using centroidal voronoi tessellations to scale up the multidimensional archive of phenotypic elites algorithm. IEEE Transactions on Evolutionary Computation. 2017 Aug 3;22(4):623-30.

       Format of the logfile: evals archive_size max mean median 5%_percentile, 95%_percentile
        dim_map: dimensionality of the map. Ex. % time contact w/ ground of 4 legs = 4 dims, etc
        dim_x: dimensionality of the behavior descriptor
        f: the environment which returns a fitness and behavior descriptor
    """
    # log hyperparams to wandb
    config_wandb(batch_size=params['batch_size'], max_evals=max_evals)

    # setup the parallel processing pool
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)

    # create the CVT
    c = cm.cvt(n_niches, dim_map,
              params['cvt_samples'], params['cvt_use_cache'])
    kdt = KDTree(c, leaf_size=30, metric='euclidean')
    cm.__write_centroids(c)

    archive = {} # init archive (empty)
    n_evals = 0 # number of evaluations since the beginning
    b_evals = 0 # number evaluation since the last dump

    # main loop
    while (n_evals < max_evals):
        start_time = time.time()
        to_evaluate = []
        # random initialization
        if len(archive) <= params['random_init'] * n_niches:
            for i in range(0, params['random_init_batch']):
                x = np.random.uniform(low=params['min'], high=params['max'], size=dim_x)
                to_evaluate += [(x, f)]
        else:  # variation/selection loop
            keys = list(archive.keys())
            # we select all the parents at the same time because randint is slow
            rand1 = np.random.randint(len(keys), size=params['batch_size'])
            rand2 = np.random.randint(len(keys), size=params['batch_size'])
            for n in range(0, params['batch_size']):
                # parent selection
                x = archive[keys[rand1[n]]]
                y = archive[keys[rand2[n]]]
                # copy & add variation
                z = variation_operator(x.x, y.x, params)
                to_evaluate += [(z, f)]
        # evaluation of the fitness for to_evaluate
        s_list = cm.parallel_eval(__evaluate, to_evaluate, pool, params)
        # natural selection
        for s in s_list:
            __add_to_archive(s, s.desc, archive, kdt)
        # count evals
        n_evals += len(to_evaluate)
        b_evals += len(to_evaluate)
        fps = len(to_evaluate) / (time.time() - start_time)
        fps = round(fps, 1)
        # write archive
        if b_evals >= params['dump_period'] and params['dump_period'] != -1:
            # print("[{}/{}]\n".format(n_evals, int(max_evals)), end=" ", flush=True)
            cm.__save_archive(archive, n_evals)
            b_evals = 0

        fit_list = np.array([x.fitness for x in archive.values()])
        # write log
        log.info(f'n_evals: {n_evals}, mean fitness: {np.mean(fit_list)}, median fitness: {np.median(fit_list)}, \
            5th percentile: {np.percentile(fit_list, 5)}, 95th percentile: {np.percentile(fit_list, 95)}')
        log.debug(f'FPS: {fps}')
        wandb.log({
            "evals": n_evals,
            "mean fitness": np.mean(fit_list),
            "median fitness": np.median(fit_list),
            "5th percentile": np.percentile(fit_list, 5),
            "95th percentile": np.percentile(fit_list, 95),
            "fps": fps
        })
    cm.__save_archive(archive, n_evals, file_name, save_path)
    return archive


class Individual(object):
    _ids = count(0)

    def __init__(self, genotype, phenotype, fitness, centroid=None):
        """
        A single agent
        param genotype:  The parameters that produced the behavior. I.e. neural network, etc.
        param phenotype: the resultant behavior i.e. the behavioral descriptor
        param fitness: the fitness of the model. In the case of a neural network, this is the total accumulated rewards
        param centroid: the closest CVT generator (a behavior) to the behavior that this individual exhibits
        """
        phenotype.id = next(self._ids)
        Individual.current_id = phenotype.id  # TODO: not sure what this is for
        self.genotype = genotype
        self.phenotype = phenotype
        self.fitness = fitness
        self.centroid = centroid
        self.novelty = None
