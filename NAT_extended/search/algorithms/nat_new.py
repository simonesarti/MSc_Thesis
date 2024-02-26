import copy
import json
import logging
import os
import time

import numpy as np
from pymoo.factory import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from NAT_extended.search.algorithms.nat import NAT
from NAT_extended.search.algorithms.utils import (
    reference_direction_survival,
    rank_n_crowding_survival,
    setup_logger,
    HighTradeoffPoints
)
from NAT_extended.search.surrogate_models import SurrogateModel
from NAT_extended.search.surrogate_models.utils import get_correlation


class NATNew(NAT):

    def __init__(
            self,
            search_space,
            evaluator,
            trainer,  # a trainer class for fine-tuning supernet, SupernetTrainer
            objs='acc&flops',
            surrogate='lgb',  # surrogate model method
            n_doe=300,  # design of experiment points, i.e., number of initial randomly sampled points
            n_gen=25,  # number of high-fidelity evaluations per generation/iteration
            max_gens=30,  # maximum number of generations/iterations to search
            topN=150,  # top N architectures from archive used to estimate distribution
            ft_epochs_gen=5,  # number of epochs for fine-tuning supernet in each generation
            save_path='.tmp',  # path to the folder for saving stats
            num_subnets=4,  # number of subnets spanning the Pareto front that you would like find
            resume_arch=None,  # path to an architecture file to resume search
            resume_ckpt=None,  # path to checkpoint file to resume search
            n_cores=1
    ):

        super(NATNew, self).__init__(
            search_space,
            evaluator,
            trainer,
            objs,
            surrogate,
            n_doe,
            n_gen,
            max_gens,
            topN,
            ft_epochs_gen,
            save_path,
            num_subnets,
            resume_arch,
            resume_ckpt,
            n_cores
        )

        self.repair_mode = "fill"  # or "push"
        self.distr_err_metric = "sse"

        self.preprocess_archive = True
        self.preprocess_archive_mult = 10



    def setup_exp(self):
        # create the save dir
        experiment_folder_name = "NATNew-{}-{}-{}-n_doe@{}-n_gen@{}-max_gens@{}-topN@{}-ft_epochs_gen@{}".format(
            self.search_space.name,
            self.objs.replace("&", "+"),  # to avoid problems while opening folders via cli
            self.surrogate,
            self.n_doe,
            self.n_gen,
            self.max_gens,
            self.topN,
            self.ft_epochs_gen
        )

        self.save_path = os.path.join(self.save_path, experiment_folder_name)
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'supernet'), exist_ok=True)  # create folder to save supernet ckpts

        # setup logger
        self.logger = logging.getLogger()
        setup_logger(self.save_path)

    def populate_archive(self):
        if self.resume_arch is not None:  # initialize archive from the point exp left off, given json file of archs
            archive = json.load(open(self.resume_arch, 'r'))
        else:  # initialize the archive with archs sampled from the search space
            archive = self.initialize_archive()
        return archive

    def initialize_archive(self):

        """
        An initialization method by sampling from search space, such that the architecture space
        is (almost) uniform. In NAT is the archive is uniform in the search space.

        :return: [
            {"arch":{"r":64, "w":1.2, ... }, "acc": 53, "params": 123, "flops": 73, "latency": 62 },
            {"arch":{decoded subnet dictionary}, "acc": 32, "params": 47, "flops": 12, "latency": 65 },
            ...]
        """

        # create an empty archive to store the architectures
        archive = []

        # sample the initial architectures from the search space
        self.search_space.set_repair_mode(self.repair_mode)

        # decide how many architectures to sample
        to_sample = self.pop_size
        if self.preprocess_archive:
            to_sample = to_sample * self.preprocess_archive_mult
        to_sample = to_sample - 2

        archs = self.search_space.sample(n_samples=to_sample, probabilities=self.search_space.initial_probs)

        # evaluate initial population in terms of the problem objectives
        stats_dict = self._eval(archs)

        # store the evaluated architectures in the archive
        for arch, stats in zip(archs, stats_dict):
            archive.append({'arch': arch, **stats})

        if self.preprocess_archive:
            archive = self.new_survival(archive, n_survive=self.pop_size-2, return_encodings=False)

        # add the lower and upper bound architectures to improve diversity among individuals
        lb_ub = self.search_space.decode([np.array(self.search_space.lb), np.array(self.search_space.ub)])
        lb_ub_stats_dict = self._eval(lb_ub)
        for arch, stats in zip(lb_ub, lb_ub_stats_dict):
            archive.append({'arch': arch, **stats})

        return archive

    def new_survival(self, archive, n_survive, return_encodings):
        # select the best n_survive architectures from the archive (how depends on number of objectives)

        X = self.search_space.encode([m['arch'] for m in archive])
        F = np.array(self.get_attributes(archive, _attr=self.objs.replace('acc', 'err')))

        n_obj = F.shape[1]
        if n_obj < 2:
            indexes = np.argsort(F[:, 0])[:n_survive]
            survived_encodings = X[indexes]

        elif n_obj > 2:
            ref_dirs = get_reference_directions("energy", n_obj, n_survive, seed=42)
            survived_encodings = reference_direction_survival(ref_dirs, X, F, n_survive=n_survive)

        else:
            survived_encodings = rank_n_crowding_survival(X, F, n_survive=n_survive)

        if return_encodings:
            return survived_encodings
        else:
            survived_archive = []
            for arch_idx, arch_enc in enumerate(X):
                if np.any(np.all(arch_enc == survived_encodings, axis=1)):
                    survived_archive.append(archive[arch_idx])
            return survived_archive

    ###########################################################################################################

    def search(self):

        # ----------------------- initialization ----------------------- #

        # creat folder to save experiment and setup logger
        self.setup_exp()

        it_start = time.time()  # time counter

        # fill the archive with architectures
        archive = self.populate_archive()

        # setup reference point and calculate hypervolume
        # ref_point value is [max_obj1, max_obj2, ...] from stats of architectures archive (use err instead of acc)
        if self.ref_pt is None:
            self.ref_pt = np.max(self.get_attributes(archive, _attr=self.objs.replace('acc', 'err')), axis=0)

        if self.resume_arch is None:    # save iteration 0 only if training is not resuming
            hypervolume = self._calc_hv(archive, self.ref_pt)
            minutes_elapsed = (time.time() - it_start) / 60
            self.logger.info(f"Iter 0: hv = {hypervolume:.4f}, time elapsed = {minutes_elapsed:.2f} mins")
            self.save_iteration("iter_0", archive)  # dump the initial population
            start_it = 1
        else:   # set correct starting iteration when resuming
            arch_path = os.path.normpath(self.resume_arch)
            split_path = arch_path.split(os.sep)
            start_it = 1 + int(split_path[-2].replace("iter_", ""))

        # ----------------------- main search loop ----------------------- #
        for it in range(start_it, self.max_gens + 1):
            it_start = time.time()  # time counter

            # construct error predictor surrogate model from archive dict
            print("fitting the error predictor")
            err_predictor = self._fit_predictors(archive)

            # construct an auxiliary problem of surrogate objectives and
            # search for the next set of candidates for high-fidelity evaluation
            candidates = self._next(archive, err_predictor)

            # high-fidelity evaluate the selected candidates
            print("evaluating the performances of the selected candidates")
            stats_dict = self._eval(candidates)

            # evaluate the performance of mIoU predictor, see how it performs using correlation measures
            print("evaluating the performances of the error predictor using true errors of the candidate architectures")
            err_pred = SurrogateModel.predict(
                model=err_predictor,
                inputs=self.search_space.features(self.search_space.encode(candidates))
            )
            err_rmse, err_r, err_rho, err_tau = get_correlation(err_pred, [100 - stat['acc'] for stat in stats_dict])

            # add the evaluated subnets to archive and keep the best n_doe
            print("adding the candidate architectures to the archive")
            for cand, stats in zip(candidates, stats_dict):
                archive.append({'arch': cand, **stats})
            archive = self.new_survival(archive, n_survive=self.n_doe, return_encodings=False)

            # estimate optimal architecture distribution from archs in the archive
            print("estimate optimal architecture distribution from archs in the archive")
            distributions = self._model_distribution(archive, verbose=False, distr_err_metric=self.distr_err_metric)  # set False to save screen

            # fine-tune supernet for some epochs
            print("fine-tune supernet")
            self.trainer.distributions = distributions
            self._adapt_supernet()

            # re-evaluate all archs in archive, because network weights have been updated
            print("update measurements for the architectures in the archive")
            archive = self._update_archive(archive)

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

        test_nd_archive = copy.deepcopy(nd_archive)
        self.test(test_nd_archive, "non_dominated_subnets")

        # dump num_subnets_to_report high-tradeoff architectures
        # (only meaningful for at least 2 objectives)
        if self.n_objs >= 2:
            nd_F = np.array(self.get_attributes(nd_archive, _attr=self.objs.replace('acc', 'err')))
            selected = HighTradeoffPoints(n_survive=self.num_subnets_to_report).do(nd_F)
            htp_archive = [nd_archive[i] for i in selected]
            self.save_subnets("high_tradeoff_subnets", htp_archive)

            test_htp_archive = copy.deepcopy(htp_archive)
            self.test(test_htp_archive, "high_tradeoff_subnets")

        # log predictor values through the iterations
        self.log_predictor()
