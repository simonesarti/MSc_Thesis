# import sys
# sys.path.insert(0, './')

import json
import logging
import os
import time

import numpy as np
from pymoo.core.problem import Problem
from pymoo.factory import get_algorithm, get_sampling, get_crossover, get_mutation, get_reference_directions
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from NAT_extended.search.algorithms.evo_nas import EvoNAS
from NAT_extended.search.algorithms.utils import (
    MySampling,
    BinaryCrossover,
    MyMutation,
    setup_logger,
    HighTradeoffPoints
)
from NAT_extended.search.surrogate_models import SurrogateModel
from NAT_extended.search.surrogate_models.utils import get_correlation

__all__ = ['MSuNAS']


class AuxiliarySingleLevelProblem(Problem):
    """ The optimization problem for finding the next N candidate architectures """

    def __init__(
        self,
        search_space,
        evaluator,
        err_predictor: SurrogateModel,
        objs='acc&flops',  # objectives to be optimized,
    ):
        super(AuxiliarySingleLevelProblem, self).__init__(
            n_var=search_space.n_var,
            n_obj=len(objs.split('&')),
            n_constr=0,
            xl=search_space.lb,
            xu=search_space.ub,
            type_var=np.int
        )

        self.search_space = search_space
        self.evaluator = evaluator
        self.err_predictor = err_predictor  # already trained predictor
        self.objs = objs

    def _evaluate(self, x, out, *args, **kwargs):
        # x is set/list of encoded architecture strings

        # use surrogate model to predict error
        features = self.search_space.features(x)
        errors = SurrogateModel.predict(self.err_predictor, features).reshape(-1, 1)
        # ==> estimated_errors = np.array [[e1],[e2], ... ]

        if self.n_obj > 1:  # exactly (high-fidelity) measure other (cheap) objectives, no err/acc
            # get dictionary representation of the architectures in x
            archs = self.search_space.decode(x)
            # for each architecture, compute list of dictionaries with performance on problem objectives (no acc)
            other_objs_stats = self.evaluator.evaluate(archs, objs=self.objs.replace('acc', ''), print_progress=False)
            # transform result into list of lists (each with the values for one architecture)
            other_objs = np.array([list(stats.values()) for stats in other_objs_stats])
            # ==> other_objs = np.array [[obj1.1, obj2.1],[obj1.2, obj2.2], ... ]
            out["F"] = np.column_stack((errors, other_objs))
            # ==> out["F"] = np.array [[e1, obj1.1, obj2.1],[e2, obj1.2, obj2.2], ... ]
        else:
            out["F"] = errors   # estimated with predictor


class SubsetSelectionProblem(Problem):
    """ select a subset to diversify the pareto front """
    def __init__(self, candidates, archive, K):

        super().__init__(
            n_var=len(candidates),
            n_obj=1,
            n_constr=1,
            xl=0,
            xu=1,
            type_var=np.bool
        )

        # todo: make sure inputs "candidates" and "archive" are [N, M] matrix, where N is pop_size, M is n_obj
        self.archive = archive  # objs array for non-dominated archs in the archive
        self.candidates = candidates    # objs array for new solutions
        self.n_max = K


    def _evaluate(self, x, out, *args, **kwargs):
        f = np.full((x.shape[0], 1), np.nan)
        g = np.full((x.shape[0], 1), np.nan)

        for i, _x in enumerate(x):
            # s, p = stats.kstest(np.concatenate((self.archive, self.candidates[_x])), 'uniform')
            # append selected candidates to archive then sort
            tmps = []
            for j in range(self.archive.shape[1]):
                tmp = np.sort(np.concatenate((self.archive[:, j], self.candidates[_x, j])))
                tmps.append(np.std(np.diff(tmp)))
            f[i, 0] = np.max(tmps)

            # f[i, 0] = s
            # we penalize if the number of selected candidates is not exactly K
            # g[i, 0] = (self.n_max - np.sum(_x)) ** 2
            g[i, 0] = np.sum(_x) - self.n_max  # as long as the selected individual is less than K

        out["F"] = f
        out["G"] = g


class MSuNAS(EvoNAS):
    """
    NSGANetV2: https://arxiv.org/abs/2007.10396
    """
    def __init__(
        self,
        search_space,
        evaluator,
        objs='acc&flops',
        surrogate='lgb',  # surrogate model method
        n_doe=100,  # design of experiment points, i.e., number of initial (usually randomly sampled) points
        n_gen=8,  # number of high-fidelity evaluations per generation/iteration
        max_gens=30,  # maximum number of generations/iterations to search
        save_path='.tmp',   # path to the folder for saving stats
        num_subnets=4,  # number of subnets spanning the Pareto front that you would like find
        resume_arch=None,  # path to an architecture file to resume search
        resume_ckpt=None,  # path to checkpoint file to resume search
        n_cores=1
    ):

        super().__init__(search_space, evaluator, objs, pop_size=n_doe, max_gens=max_gens)

        self.surrogate = surrogate
        self.n_doe = n_doe
        self.n_gen = n_gen
        self.num_subnets_to_report = num_subnets
        self.resume_arch = resume_arch
        self.resume_ckpt = resume_ckpt
        self.ref_pt = None
        self.save_path = save_path
        self.logger = None  # a placeholder
        self.n_cores = n_cores

    def _sample_initial_population(self, sample_size):
        """
        A generic initialization method by uniform sampling from search space,
        note that uniform sampling from search space (x or genotype space -> "encoded string of values")
        does NOT imply uniformity in architecture space (phenotype -> readable/decoded representation of net, dict).
        """

        archs = self.search_space.sample(sample_size - 2)
        # add the lower and upper bound architectures for improving diversity among individuals
        archs.extend(self.search_space.decode([np.array(self.search_space.lb), np.array(self.search_space.ub)]))

        # returns list of randomly sampled ->decoded<- architectures
        return archs

    def _fit_predictors(self, archive):
        self.logger.info("fitting {} model for accuracy/error ...".format(self.surrogate))

        data = self.get_attributes(archive, _attr='arch&err')   # [[dec_arch,err], [dec_arch,err], ...]

        # extract features from the encoded version of the architectures in the archive
        features = self.search_space.features(self.search_space.encode([d[0] for d in data]))
        # create array with corresponding target error values
        err_targets = np.array([d[1] for d in data])

        # obtain trained model to predict error given the encoded representation of a net
        err_predictor = SurrogateModel(self.surrogate).fit(features, err_targets, ensemble=True, n_cores=self.n_cores)

        return err_predictor

    @staticmethod
    def select_solver(
        n_obj,
        _pop_size=100,
        _crx_prob=0.9,  # crossover probability
        _mut_eta=1.0,  # polynomial mutation hyperparameter eta
        _seed=42  # random seed for riesz energy
    ):
        # define operators
        sampling = get_sampling('int_lhs')
        crossover = get_crossover('int_two_point', prob=_crx_prob)
        mutation = get_mutation('int_pm', eta=_mut_eta)

        if n_obj < 2:
            # use ga, de, pso, etc.
            ea_method = get_algorithm(
                "ga",
                pop_size=_pop_size,
                sampling=sampling,
                crossover=crossover,
                mutation=mutation,
                eliminate_duplicates=True
            )

        elif n_obj > 2:
            # use NSGA-III
            # create the reference directions to be used for the optimization
            # # use this if you are familiar with many-obj optimization
            # ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
            ref_dirs = get_reference_directions("energy", n_obj, _pop_size, seed=_seed)
            ea_method = get_algorithm(
                'nsga3',
                pop_size=_pop_size,
                ref_dirs=ref_dirs,
                sampling=sampling,
                crossover=crossover,
                mutation=mutation,
                eliminate_duplicates=True
            )

        else:
            # use NSGA-II, MOEA/D
            ea_method = get_algorithm(
                "nsga2",
                pop_size=_pop_size,
                sampling=sampling,
                crossover=crossover,
                mutation=mutation,
                eliminate_duplicates=True
            )

        return ea_method

    def subset_selection(self, pop, archive, K):
        # get non-dominated archs from archive
        F = np.array(self.get_attributes(archive, _attr=self.objs.replace('acc', 'err')))
        # F = [[err,obj1,obj2], [err, obj1, obj2], ...] for archs in the archive
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        # front is np.array with the indexes of the non-dominated archs given the objective arrays in F

        if not self.objs == "acc":  # if accuracy is not the only objective
            # select based on the cheap objectives, i.e., Params, FLOPs, etc.
            ssp_candidates = pop.get("F")[:, 1:]     # arrays of objs corresponding to new population, exclude first (err)
            ssp_archive = F[front, 1:]  # arrays of objs corresponding to non-dominated architectures, excluded first
            subset_problem = SubsetSelectionProblem(ssp_candidates, ssp_archive, K)

            # define a solver
            ea_method = get_algorithm(
                'ga',
                pop_size=500,
                sampling=MySampling(),
                crossover=BinaryCrossover(),
                mutation=MyMutation(),
                eliminate_duplicates=True
            )

            # start solving
            res = minimize(subset_problem, ea_method, termination=('n_gen', 200), verbose=True)  # set verbos=False to save screen
            # in case the number of solutions selected is less than K, add to selected indexes the ones of archs with the
            # best first objective (estim_err), which was excluded from the problem above
            if np.sum(res.X) < K:
                for idx in np.argsort(pop.get("F")[:, 0]):
                    res.X[idx] = True
                    if np.sum(res.X) >= K:
                        break
            return res.X

        else:   # sort candidates by error, return the ones minimizing it
            ssp_candidates = pop.get("F")[:, 0]
            subset_array = np.full(pop.get("F").shape[0], False)
            for idx in np.argsort(ssp_candidates):
                subset_array[idx] = True
                if np.sum(subset_array) >= K:
                    break
            return subset_array

    def _next(self, archive, err_predictor):
        # initialize the candidate finding optimization problem:
        # solutions will be searched through the whole problem search space and result of
        # optimization depends on the error predictor given at each _next iteration
        problem = AuxiliarySingleLevelProblem(self.search_space, self.evaluator, err_predictor, self.objs)

        # this problem is a regular discrete-variable single-/multi-/many-objective problem
        # which can be exhaustively searched by regular EMO algorithms such as rGA, NSGA-II/-III, MOEA/D, etc.
        ea_method = self.select_solver(problem.n_obj)

        # applies the algorithm to solve the problem by minimizing the objectives
        # res.pop contains the new population, archs minimizing the objectives [[estim_err,obj1],[...]] = out["F"]
        print("selecting the new population")
        res = minimize(problem, ea_method, termination=('n_gen', 20), verbose=True)  # set verbose=False to save screen

        # check resulting population (the encodings of the solutions/archs found) and check if they are already present
        # in the archive, to eliminate any already evaluated subnets to be re-evaluated
        not_duplicate = np.logical_not(
            [x in [x[0] for x in self.get_attributes(archive, _attr='arch')]
             for x in self.search_space.decode(res.pop.get("X"))])

        # form a subset selection problem to short list K from pop_size
        # population is the set of new solutions that are not already in the archive
        print("selecting the new candidate architectures")
        indices = self.subset_selection(pop=res.pop[not_duplicate], archive=archive, K=self.n_gen)

        # candidates are the decoded, non-duplicate archs obtained from the objective minimization problem,
        # of which only K are kept, their indices being given by the solution of the subset selection problem
        candidates = self.search_space.decode(res.pop[not_duplicate][indices].get("X"))

        return candidates

    def search(self):
        # ----------------------- setup ----------------------- #
        # create the save dir and setup logger
        self.save_path = os.path.join(
            self.save_path,
            self.search_space.name + "NSGANetV2-{}-{}-n_doe@{}-n_gen@{}-max_gens@{}"
            .format(self.objs.replace("&", "+"), self.surrogate, self.n_doe, self.n_gen, self.max_gens)
        )

        os.makedirs(self.save_path, exist_ok=True)
        self.logger = logging.getLogger()
        setup_logger(self.save_path)

        # ----------------------- initialization ----------------------- #
        it_start = time.time()  # time counter
        if self.resume_arch:
            archive = json.load(open(self.resume_arch, 'r'))
        else:
            archive = self.initialization()     # is list of dicts (arch and corresponding stats)

        # setup reference point for calculating hypervolume
        if self.ref_pt is None:
            self.ref_pt = np.max(self.get_attributes(archive, _attr=self.objs.replace('acc', 'err')), axis=0)
            # ref_point value is [max_obj1, max_obj2, ...] from stats of the architectures in the archive,
            # using err instead of acc. ==> (architecture is NOT an objective)

        self.logger.info("Iter 0: hv = {:.4f}, time elapsed = {:.2f} mins".format(
            self._calc_hv(archive, self.ref_pt),    # hypervolume calculation
            (time.time() - it_start) / 60)
        )

        self.save_iteration("iter_0", archive)  # dump the initial population

        # ----------------------- main search loop ----------------------- #
        for it in range(1, self.max_gens + 1):
            it_start = time.time()  # time counter

            # construct error predictor surrogate model from archive dict
            # (as architectures are added to the archive after each iteration, it should get better)
            err_predictor = self._fit_predictors(archive)

            # construct an auxiliary problem of surrogate objectives and
            # search for the next set of candidates for high-fidelity evaluation
            candidates = self._next(archive, err_predictor)

            # high-fidelity evaluate the new selected candidates (lower level)
            stats_dict = self._eval(candidates)

            # evaluate the performance of mIoU predictor, see how it performs using correlation measures
            err_pred = SurrogateModel.predict(
                model=err_predictor,
                inputs=self.search_space.features(self.search_space.encode(candidates))
            )
            err_rmse, err_r, err_rho, err_tau = get_correlation(err_pred, [100 - stat['acc'] for stat in stats_dict])

            # add the evaluated subnets to archive
            for cand, stats in zip(candidates, stats_dict):
                archive.append({'arch': cand, **stats})

            # print iteration-wise statistics
            hv = self._calc_hv(archive, self.ref_pt)
            iter_time_elapsed = (time.time() - it_start) / 60
            self.logger.info("Iter {}: hv = {:.4f}, time elapsed = {:.2f} mins".format(it, hv, iter_time_elapsed))
            self.logger.info("Surrogate model {} performance:".format(self.surrogate))
            self.logger.info("For predicting mIoU: RMSE = {:.4f}, Spearman's Rho = {:.4f}, "
                             "Kendall’s Tau = {:.4f}".format(err_rmse, err_rho, err_tau))

            predictor_stats = {
                "Kendall Tau": err_tau,
                "Spearman Rho": err_rho,
                "RMSE": err_rmse
            }
            meta_stats = {
                'iteration': it,
                'time_mins': iter_time_elapsed,
                'hv': hv,
                'predictor': predictor_stats
            }
            self.save_iteration("iter_{}".format(it), archive, meta_stats)  # dump the current iteration

        # ----------------------- report search result ----------------------- #
        # dump non-dominated architectures from the archive first
        F = np.array(self.get_attributes(archive, _attr=self.objs.replace('acc', 'err')))
        nd_front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        nd_archive = [archive[idx] for idx in nd_front]
        self.save_subnets("non_dominated_subnets", nd_archive)

        # select a subset from non-dominated set in case further fine-tuning
        if self.n_objs >= 2:
            nd_F = np.array(self.get_attributes(nd_archive, _attr=self.objs.replace('acc', 'err')))
            selected = HighTradeoffPoints(n_survive=self.num_subnets_to_report).do(nd_F)
            self.save_subnets("high_tradeoff_subnets", [nd_archive[i] for i in selected])

    def save_iteration(self, _save_dir, archive, meta_stats=None):
        save_dir = os.path.join(self.save_path, _save_dir)
        os.makedirs(save_dir, exist_ok=True)
        json.dump(archive, open(os.path.join(save_dir, 'archive.json'), 'w'), indent=4)
        if meta_stats:
            json.dump(meta_stats, open(os.path.join(save_dir, 'stats.json'), 'w'), indent=4)

    def save_subnets(self, _save_dir, archive):
        save_dir = os.path.join(self.save_path, _save_dir)
        os.makedirs(save_dir, exist_ok=True)
        for i, subnet in enumerate(archive):
            json.dump(subnet, open(os.path.join(save_dir, 'subnet_{}.json'.format(i + 1)), 'w'), indent=4)


