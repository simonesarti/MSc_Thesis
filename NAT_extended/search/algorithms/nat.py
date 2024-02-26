import copy
import json
import logging
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from pymoo.factory import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from NAT_extended.search.algorithms.msunas import MSuNAS
from NAT_extended.search.algorithms.utils import distribution_estimation, multiprocessing_distribution_estimation
from NAT_extended.search.algorithms.utils import (
    reference_direction_survival,
    rank_n_crowding_survival,
    setup_logger,
    HighTradeoffPoints
)
from NAT_extended.search.surrogate_models import SurrogateModel
from NAT_extended.search.surrogate_models.utils import get_correlation


def create_scatter_plot_2D(x_name, y_name, x_values, y_values, plots_folder):
    plt.scatter(x_values, y_values)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.grid()
    x_name = x_name[:-4].replace(" ", "")
    y_name = y_name[:-4].replace(" ", "")
    plt.savefig(os.path.join(plots_folder, f"{x_name}_vs_{y_name}.png"))
    plt.clf()


def create_scatter_plot_3D(x_name, y_name, z_name, x_values, y_values, z_values, plots_folder):

    # Creating figure
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")

    # Creating plot
    ax.scatter3D(x_values, y_values, z_values, color="blue")

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_zlabel(z_name)

    x_name = x_name[:-4].replace(" ", "")
    y_name = y_name[:-4].replace(" ", "")
    z_name = z_name[:-4].replace(" ", "")

    plt.savefig(os.path.join(plots_folder, f"{x_name}_vs_{y_name}_vs_{z_name}.png"))
    plt.clf()


def plot_objs(archive, save_dir):

    plots_folder = os.path.join(save_dir, "plots")
    os.makedirs(plots_folder, exist_ok=True)

    top1 = np.array([arch["test"]["top1"] for arch in archive])
    params = np.array([arch["net_stat"]["params"] for arch in archive])
    flops = np.array([arch["net_stat"]["flops"] for arch in archive])
    latency = np.array([arch["net_stat"]["latency"] for arch in archive])

    create_scatter_plot_2D("params (M)", "top1 (%)", params, top1, plots_folder)
    create_scatter_plot_2D("flops (M)", "top1 (%)", flops, top1, plots_folder)
    create_scatter_plot_2D("latency (ms)", "top1 (%)", latency, top1, plots_folder)

    create_scatter_plot_2D("params (M)", "flops (M)", params, flops, plots_folder)
    create_scatter_plot_2D("params (M)", "latency (ms)", params, latency, plots_folder)
    create_scatter_plot_2D("flops (M)", "latency (ms)", flops, latency, plots_folder)

    create_scatter_plot_3D("params (M)", "flops (M)", "top1 (%)", params, flops, top1, plots_folder)
    create_scatter_plot_3D("params (M)", "latency (ms)", "top1 (%)", params, latency, top1, plots_folder)
    create_scatter_plot_3D("flops (M)", "latency (ms)", "top1 (%)", flops, latency, top1, plots_folder)


class NAT(MSuNAS):
    """
    Neural Architecture Transfer: https://arxiv.org/pdf/2005.05859.pdf
    """

    def __init__(
        self,
        search_space,
        evaluator,
        trainer,  # a trainer class for fine-tuning supernet, SupernetTrainer
        objs='acc&flops',
        surrogate='lgb',  # surrogate model method
        n_doe=100,  # design of experiment points, i.e., number of initial (usually randomly sampled) points
        n_gen=8,  # number of high-fidelity evaluations per generation/iteration
        max_gens=30,  # maximum number of generations/iterations to search
        topN=150,  # top N architectures from archive used to estimate distribution
        ft_epochs_gen=5,  # number of epochs for fine-tuning supernet in each generation
        save_path='.tmp',  # path to the folder for saving stats
        num_subnets=4,  # number of subnets spanning the Pareto front that you would like find
        resume_arch=None,  # path to an architecture file to resume search
        resume_ckpt=None,    # path to checkpoint file to resume search
        n_cores=1
    ):

        super().__init__(
            search_space,
            evaluator,
            objs,
            surrogate,
            n_doe,
            n_gen,
            max_gens,
            save_path,
            num_subnets,
            resume_arch,
            resume_ckpt,
            n_cores
        )

        self.topN = topN
        self.ft_epochs_gen = ft_epochs_gen

        self.best_acc = None
        self.trainer = trainer
        self.cur_epoch = 0  # a counter to keep track of how epochs of training applied to supernet

        assert (resume_arch is not None and resume_ckpt is not None) or (resume_arch is None and resume_ckpt is None), \
            "both resume_arch and resume_ckpt must be either both specified or both unspecified"

        if self.resume_ckpt is not None:
            self.restore_training_state()

    def restore_training_state(self):
        sd = torch.load(self.resume_ckpt, map_location='cpu')
        self.cur_epoch = sd["epoch"] + 1

        self.trainer.cur_epoch = self.cur_epoch
        trainer_models_dicts = [sd["model_w1.0_state_dict"], sd["model_w1.2_state_dict"]]
        self.trainer.supernet.load_state_dict(trainer_models_dicts)
        self.trainer.optimizers[0].load_state_dict(sd["optimizer_w1.0_state_dict"]),
        self.trainer.optimizers[1].load_state_dict(sd["optimizer_w1.2_state_dict"])

    def survival(self, archive):
        # select the best topN architectures from the archive (how depends on number of objectives)
        # and return their encoded form

        X = self.search_space.encode([m['arch'] for m in archive])
        F = np.array(self.get_attributes(archive, _attr=self.objs.replace('acc', 'err')))

        n_obj = F.shape[1]
        if n_obj < 2:
            indexes = np.argsort(F[:, 0])[:self.topN]
            return X[indexes]

        elif n_obj > 2:
            ref_dirs = get_reference_directions("energy", n_obj, self.topN, seed=42)
            return reference_direction_survival(ref_dirs, X, F, n_survive=self.topN)

        else:
            return rank_n_crowding_survival(X, F, n_survive=self.topN)

    def _model_distribution(self, archive, verbose=True, distr_err_metric="sse"):
        # estimate the optimal distribution per architectural variable from archive

        # filter top-N architectures from archive first
        X = self.survival(archive)

        # one distribution for each encoded value in the encodings,

        if self.n_cores == 1:    # >> single process version
            distributions = []
            for j in range(X.shape[1]):
                distributions.append(distribution_estimation(X[:, j], verbose, distr_err_metric))
            return distributions

        else:   # >> multiprocess version
            encodings_list = [X[:, j] for j in range(X.shape[1])]
            distributions = multiprocessing_distribution_estimation(encodings_list, self.n_cores, verbose, distr_err_metric)
            return distributions

    """ validates all epochs
    def _adapt_supernet(self):
        for epoch in range(self.cur_epoch, self.cur_epoch + self.ft_epochs_gen):

            # self.trainer.logger.info('epoch {:d} lr {:.2e}'.format(epoch, self.trainer.schedulers[0].get_lr()[0]))
            self.logger.info('epoch {:d} lr {:.2e}'.format(epoch + 1, self.trainer.schedulers[0].get_epoch_values(epoch)[0]))

            train_loss, (train_top1, train_top5) = self.trainer.train_one_epoch(epoch)

            print_str = "TRAIN: loss={:.4f}, top1={:.4f}, top5={:.4f} -- ".format(train_loss, train_top1, train_top5)

            save_dict = {
                'epoch': epoch,
                'model_w1.0_state_dict': self.trainer.supernet.engine[0].state_dict(),
                'model_w1.2_state_dict': self.trainer.supernet.engine[1].state_dict(),
                'optimizer_w1.0_state_dict': self.trainer.optimizers[0].state_dict(),
                'optimizer_w1.2_state_dict': self.trainer.optimizers[1].state_dict()
            }

            valid_loss, (valid_acc1, valid_acc5) = self.trainer.validate(epoch)
            print_str += "VALIDATION: loss={:.4f}, top1={:.4f}, top5={:.4f}".format(valid_loss, valid_acc1, valid_acc5)

            self.trainer.logger.info(print_str)

            for scheduler in self.trainer.schedulers:
                scheduler.step(epoch + 1)

            torch.save(save_dict, os.path.join(self.save_path, 'supernet', 'checkpoint.pth.tar'))

            if self.best_acc is None or self.best_acc < valid_acc1:
                self.best_acc = valid_acc1
                torch.save(save_dict, os.path.join(self.save_path, 'supernet', 'model_best.pth.tar'))

            self.cur_epoch += 1

        torch.save(save_dict, os.path.join(self.save_path,
                                           'supernet', 'checkpoint_epoch@{}.pth.tar'.format(self.cur_epoch)))
    """
    def _adapt_supernet(self):  # validates only last epoch, saves time
        for epoch in range(self.cur_epoch, self.cur_epoch + self.ft_epochs_gen):

            # self.trainer.logger.info('epoch {:d} lr {:.2e}'.format(epoch, self.trainer.schedulers[0].get_lr()[0]))
            self.logger.info('epoch {:d} lr {:.2e}'.format(epoch + 1, self.trainer.schedulers[0].get_epoch_values(epoch)[0]))

            train_loss, (train_top1, train_top5) = self.trainer.train_one_epoch(epoch)

            print_str = "TRAIN: loss={:.4f}, top1={:.4f}, top5={:.4f} -- ".format(train_loss, train_top1, train_top5)
            self.trainer.logger.info(print_str)

            save_dict = {
                'epoch': epoch,
                'model_w1.0_state_dict': self.trainer.supernet.engine[0].state_dict(),
                'model_w1.2_state_dict': self.trainer.supernet.engine[1].state_dict(),
                'optimizer_w1.0_state_dict': self.trainer.optimizers[0].state_dict(),
                'optimizer_w1.2_state_dict': self.trainer.optimizers[1].state_dict()
            }
            for scheduler in self.trainer.schedulers:
                scheduler.step(epoch + 1)

            torch.save(save_dict, os.path.join(self.save_path, 'supernet', 'checkpoint.pth.tar'))

            self.cur_epoch += 1

        valid_loss, (valid_acc1, valid_acc5) = self.trainer.validate(self.cur_epoch - 1)
        print_str = "VALIDATION: loss={:.4f}, top1={:.4f}, top5={:.4f}".format(valid_loss, valid_acc1, valid_acc5)
        self.trainer.logger.info(print_str)

        torch.save(save_dict, os.path.join(self.save_path,
                                           'supernet', 'checkpoint_epoch@{}.pth.tar'.format(self.cur_epoch)))

    def _update_archive(self, archive):
        # re-evaluate all members in archive
        archs = [m['arch'] for m in archive]
        stats_dict = self._eval(archs)  # evaluate
        return [{'arch': arch, **stats} for arch, stats in zip(archs, stats_dict)]

    def search(self):
        # ----------------------- setup ----------------------- #
        # create the save dir and setup logger
        self.save_path = os.path.join(
            self.save_path, 'NAT-' + self.search_space.name +
                            "-{}-{}-n_doe@{}-n_gen@{}-max_gens@{}-topN@{}-ft_epochs_gen@{}".format(
                                self.objs.replace("&", "+"),    # to avoid problems while opening folders via cli
                                self.surrogate,
                                self.n_doe,
                                self.n_gen,
                                self.max_gens,
                                self.topN,
                                self.ft_epochs_gen))
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'supernet'), exist_ok=True)  # make a folder to save all supernet ckpts
        self.logger = logging.getLogger()
        setup_logger(self.save_path)

        # ----------------------- initialization ----------------------- #
        it_start = time.time()  # time counter

        if self.resume_arch is not None:
            archive = json.load(open(self.resume_arch, 'r'))
            # recompute the ref point given the archive
            if self.ref_pt is None:
                self.ref_pt = np.max(self.get_attributes(archive, _attr=self.objs.replace('acc', 'err')), axis=0)
        else:
            archive = self.initialization()     # is list of dicts (arch and corresponding stats)

            # setup reference point for calculating hypervolume
            if self.ref_pt is None:
                self.ref_pt = np.max(self.get_attributes(archive, _attr=self.objs.replace('acc', 'err')), axis=0)
                # ref_point value is [max_obj1, max_obj2, ...] from stats of the architectures in the archive,
                # using err instead of acc. ==> (architecture is NOT an objective)

            self.logger.info("Iter 0: hv = {:.4f}, time elapsed = {:.2f} mins".format(
                self._calc_hv(archive, self.ref_pt), (time.time() - it_start) / 60))

            self.save_iteration("iter_0", archive)  # dump the initial population

        # ----------------------- main search loop ----------------------- #
        if self.resume_arch is not None:
            arch_path = os.path.normpath(self.resume_arch)
            split_path = arch_path.split(os.sep)
            start_it = 1 + int(split_path[-2].replace("iter_", ""))
        else:
            start_it = 1

        for it in range(start_it, self.max_gens + 1):
            it_start = time.time()  # time counter

            # construct error predictor surrogate model from archive dict
            # (as architectures are added to the archive after each iteration, it should get better)
            print("fitting the error predictor")
            err_predictor = self._fit_predictors(archive)

            # construct an auxiliary problem of surrogate objectives and
            # search for the next set of candidates for high-fidelity evaluation
            candidates = self._next(archive, err_predictor)

            # high-fidelity evaluate the selected candidates (lower level)
            print("evaluating the performances of the selected candidates")
            stats_dict = self._eval(candidates)

            # evaluate the performance of mIoU predictor, see how it performs using correlation measures
            print("evaluating the performances of the error predictor using true errors of the candidate architectures")
            err_pred = SurrogateModel.predict(
                model=err_predictor,
                inputs=self.search_space.features(self.search_space.encode(candidates))
            )
            err_rmse, err_r, err_rho, err_tau = get_correlation(err_pred, [100 - stat['acc'] for stat in stats_dict])

            # add the evaluated subnets to archive
            print("adding the candidate architectures to the archive")
            for cand, stats in zip(candidates, stats_dict):
                archive.append({'arch': cand, **stats})

            # estimate optimal architecture distribution from archs in the archive
            print("estimate optimal architecture distribution from archs in the archive")
            distributions = self._model_distribution(archive, verbose=False)

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

    def test(self, archive, save_folder):

        save_dir = os.path.join(self.save_path, save_folder)
        subnets_strings = [arch["arch"] for arch in archive]

        # create logging file
        log_file_path = os.path.join(save_dir, "test_log.txt")
        module_strings_path = os.path.join(save_dir, "module_strings.txt")

        # open log files
        log_file = open(log_file_path, "w")
        mod_str_file = open(module_strings_path, "w")

        # evaluate the subnets
        net_stats = self.evaluator.evaluate(subnets_strings, "flops&params&latency")
        data_provider = self.evaluator.data_provider

        final_archive = []
        for i, (subnet_str, net_stat) in enumerate(zip(subnets_strings, net_stats), 1):

            image_scale = subnet_str.pop("r")
            self.evaluator.supernet.set_active_subnet(**subnet_str)
            subnet = self.evaluator.supernet.get_active_subnet(preserve_weight=True)
            subnet.cuda()

            data_provider.assign_active_img_size(image_scale)
            dl = data_provider.test
            sdl = data_provider.build_sub_train_loader(data_provider.subset_size, data_provider.subset_batch_size)

            # compute top-1 accuracy
            criterion = torch.nn.CrossEntropyLoss()
            loss, top1, top5 = self.evaluator.eval_acc(subnet, dl, sdl, criterion)

            # create dictionary to report
            subnet_str["r"] = image_scale
            test_results = {
                "loss": round(loss, 4),
                "top1": round(top1, 4),
                "err1": round(100 - top1, 4),
                "top5": round(top5, 4),
                "err5": round(100 - top5, 4)
            }
            for k, v in net_stat.items():
                net_stat[k] = round(v, 4)

            logged_net = {
                "arch": subnet_str,
                "test": test_results,
                "net_stat": net_stat
            }
            final_archive.append(logged_net)

            # log network results
            log_str = f"Network in subnet_{i}.json:\n"
            log_str += json.dumps(logged_net) + "\n\n"
            log_file.write(log_str)

            # log network structure
            mod_str = f"Network in subnet_{i}.json:\n"
            mod_str += subnet.module_str
            mod_str += "\n----------------------------\n\n"
            mod_str_file.write(mod_str)

        log_file.close()
        mod_str_file.close()

        # log networks also by objective result
        self.sort_subnets_by_obj(copy.deepcopy(final_archive), save_dir)

        # create correlation plots
        plot_objs(copy.deepcopy(final_archive), save_dir)

    def sort_subnets_by_obj(self, archive, save_dir):

        sorted_file_path = os.path.join(save_dir, "sorted_by_obj.txt")
        sorted_file = open(sorted_file_path, "w")

        for i, arch in enumerate(archive, 1):
            arch["i"] = i

        # sort by accuracy
        acc_sorted_archs = sorted(archive, key=lambda d: d["test"]["top1"], reverse=True)
        _str = ">>>>>>>>>>>>>>>>>>>>>>> TOP1 <<<<<<<<<<<<<<<<<<<<<<<<<<<<\n"
        for arch in acc_sorted_archs:
            n = arch["i"]
            top1 = arch['test']['top1']
            params = arch["net_stat"]["params"]
            flops = arch["net_stat"]["flops"]
            latency = arch["net_stat"]["latency"]
            _str += f"({n}): {top1}         ---- params:{params} M, flops:{flops} M, latency: {latency} ms\n"

        # sort by params
        if "params" in self.objs:
            params_sorted_archs = sorted(archive, key=lambda d: d["net_stat"]["params"])
            _str += f"\n\n>>>>>>>>>>>>>>>>>>>>>>> PARAMS <<<<<<<<<<<<<<<<<<<<<<<<<<<<\n"
            for arch in params_sorted_archs:
                n = arch["i"]
                top1 = arch['test']['top1']
                params = arch["net_stat"]["params"]
                flops = arch["net_stat"]["flops"]
                latency = arch["net_stat"]["latency"]
                _str += f"({n}): {params} M         ---- top1:{top1}, flops:{flops} M, latency:{latency} ms\n"

        # sort by flops
        if "flops" in self.objs:
            flops_sorted_archs = sorted(archive, key=lambda d: d["net_stat"]["flops"])
            _str += f"\n\n>>>>>>>>>>>>>>>>>>>>>>> FLOPS <<<<<<<<<<<<<<<<<<<<<<<<<<<<\n"
            for arch in flops_sorted_archs:
                n = arch["i"]
                top1 = arch['test']['top1']
                params = arch["net_stat"]["params"]
                flops = arch["net_stat"]["flops"]
                latency = arch["net_stat"]["latency"]
                _str += f"({n}): {flops} M         ---- top1:{top1}, params:{params} M, latency:{latency} ms\n"

        # sort by latency
        if "latency" in self.objs:
            latency_sorted_archs = sorted(archive, key=lambda d: d["net_stat"]["latency"])
            _str += f"\n\n>>>>>>>>>>>>>>>>>>>>>>> LATENCY <<<<<<<<<<<<<<<<<<<<<<<<<<<<\n"
            for arch in latency_sorted_archs:
                n = arch["i"]
                top1 = arch['test']['top1']
                params = arch["net_stat"]["params"]
                flops = arch["net_stat"]["flops"]
                latency = arch["net_stat"]["latency"]
                _str += f"({n}): {latency} ms         ---- top1:{top1}, params:{params} M, flops:{flops} M\n"

        sorted_file.write(_str)
        sorted_file.close()

    def log_predictor(self):

        predictor_stats_folder = os.path.join(self.save_path, "predictor_stats")
        os.makedirs(predictor_stats_folder, exist_ok=True)

        rmse = []
        kendall_tau = []
        spearman_rho = []

        for i in range(1, self.max_gens+1):

            with open(os.path.join(self.save_path, f"iter_{i}", "stats.json")) as jf:
                data = json.load(jf)

            if "predictor" in data:
                rmse.append(data["predictor"]["RMSE"])
                kendall_tau.append(data["predictor"]["Kendall Tau"])
                spearman_rho.append(data["predictor"]["Spearman Rho"])
            else:
                kendall_tau.append(data["acc_tau"])

        kendall_tau_str = "; ".join(str(value) for value in kendall_tau)
        rmse_str = "; ".join(str(value) for value in rmse)
        spearman_rho_str = "; ".join(str(value) for value in spearman_rho)
        with open(os.path.join(predictor_stats_folder, "values.txt"), "w") as pvf:
            pvf.write("KENDALL TAU:" + kendall_tau_str)
            pvf.write("\n")
            pvf.write("RMSE: " + rmse_str)
            pvf.write("\n")
            pvf.write("SPEARMAN RHO: " + spearman_rho_str)

        x = np.arange(1, self.max_gens+1)

        if len(rmse) > 0:
            for i, value in enumerate(rmse):
                if math.isnan(value):
                    rmse[i] = np.nan
            rmse = np.array(rmse)
            plt.plot(x, rmse, 'o-')
            plt.savefig(os.path.join(predictor_stats_folder, f"rmse.png"))
            plt.clf()

        if len(spearman_rho) > 0:
            for i, value in enumerate(spearman_rho):
                if math.isnan(value):
                    spearman_rho[i] = np.nan
            spearman_rho = np.array(spearman_rho)
            plt.plot(x, spearman_rho, 'o-')
            plt.savefig(os.path.join(predictor_stats_folder, f"spearman_rho.png"))
            plt.clf()

        for i, value in enumerate(kendall_tau):
            if math.isnan(value):
                kendall_tau[i] = np.nan
        kendall_tau = np.array(kendall_tau)
        plt.plot(x, kendall_tau, 'o-')
        plt.savefig(os.path.join(predictor_stats_folder, f"kendall_tau.png"))
        plt.clf()
