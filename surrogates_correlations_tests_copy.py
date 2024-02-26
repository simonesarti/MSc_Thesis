import torch
import numpy as np
import time
import random
from NAT_extended.data_providers.providers_factory import get_data_provider
from NAT_extended.search.search_spaces.search_spaces_factory import get_search_space
from NAT_extended.supernets.supernets_factory import get_supernet
from NAT_extended.search.evaluators.ofa_evaluator import OFAEvaluator
from NAT_extended.search.surrogate_models import SurrogateModel
from NAT_extended.search.surrogate_models.utils import get_correlation
import json
import argparse

def load_builders(net_name, feature_encoding="one-hot"):

    base_ckpt = "running_experiment/storage/NAT/NAT_TRAIN/run/from_tiny_imagenet/to_tiny_imagenet/OFAMobileNetV3/Nexp_2/NAT-b_ofamobilenetv3_ss-acc+params-lgb-n_doe@100-n_gen@8-max_gens@30-topN@150-ft_epochs_gen@5/supernet/checkpoint_epoch@150.pth.tar"
    
    SE_D_ckpt = "running_experiment/storage/NAT/NAT_TRAIN/run/from_tiny_imagenet/to_tiny_imagenet/SE_D_OFAMobileNetV3/Nexp_2/NAT-b_ofamobilenetv3_ss-acc+params-lgb-n_doe@100-n_gen@8-max_gens@30-topN@150-ft_epochs_gen@5/supernet/checkpoint_epoch@150.pth.tar"
    
    EE_B_ckpt = "running_experiment/storage/NAT/NAT_TRAIN/run/from_tiny_imagenet/to_tiny_imagenet/EE_B_OFAMobileNetV3/Nexp_2/NAT-nd_ofamobilenetv3_ss-acc+params-lgb-n_doe@100-n_gen@8-max_gens@30-topN@150-ft_epochs_gen@5/supernet/checkpoint_epoch@150.pth.tar"
    EE_D_ckpt = "running_experiment/storage/NAT/NAT_TRAIN/run/from_tiny_imagenet/to_tiny_imagenet/EE_D_OFAMobileNetV3/Nexp_2/NAT-nd_ofamobilenetv3_ss-acc+params-lgb-n_doe@100-n_gen@8-max_gens@30-topN@150-ft_epochs_gen@5/supernet/checkpoint_epoch@150.pth.tar"
    
    SE_P_ckpt = "running_experiment/storage/NAT/NAT_TRAIN/run/from_tiny_imagenet/to_tiny_imagenet/SE_P_OFAMobileNetV3/Nexp_2/NAT-p_ofamobilenetv3_ss-acc+params-lgb-n_doe@100-n_gen@8-max_gens@30-topN@150-ft_epochs_gen@5/supernet/checkpoint_epoch@150.pth.tar"
    SE_DP_ckpt = "running_experiment/storage/NAT/NAT_TRAIN/run/from_tiny_imagenet/to_tiny_imagenet/SE_DP_OFAMobileNetV3/Nexp_2/NAT-p_ofamobilenetv3_ss-acc+params-lgb-n_doe@100-n_gen@8-max_gens@30-topN@150-ft_epochs_gen@5/supernet/checkpoint_epoch@150.pth.tar"
    
    EE_P_ckpt = "running_experiment/storage/NAT/NAT_TRAIN/run/from_tiny_imagenet/to_tiny_imagenet/EE_P_OFAMobileNetV3/Nexp_2/NAT-nd_p_ofamobilenetv3_ss-acc+params-lgb-n_doe@100-n_gen@8-max_gens@30-topN@150-ft_epochs_gen@5/supernet/checkpoint_epoch@150.pth.tar"
    EE_DP_ckpt = "running_experiment/storage/NAT/NAT_TRAIN/run/from_tiny_imagenet/to_tiny_imagenet/EE_DP_OFAMobileNetV3/Nexp_2/NAT-nd_p_ofamobilenetv3_ss-acc+params-lgb-n_doe@100-n_gen@8-max_gens@30-topN@150-ft_epochs_gen@5/supernet/checkpoint_epoch@150.pth.tar"

    data_provider = get_data_provider(
        dataset="tiny_imagenet",
        save_path=None,
        train_batch_size=128,
        test_batch_size=128,
        valid_size=0.15,
        n_worker=4,
        image_size=[48, 56, 64],
        resize_scale=0.85,
        distort_color=None,
        num_replicas=None,
        rank=None
    )

    # construct the search space
    search_space = get_search_space(
        net_name=net_name,
        search_type="ea",
        image_scale_list=[48, 56, 64],
        feature_encoding=feature_encoding
    )

    # construct the supernet
    supernet = get_supernet(
        net_name=net_name,
        n_classes=data_provider.n_classes,
        dropout_rate=0,
        search_space=search_space
    )

    evaluator = OFAEvaluator(
        supernet=supernet,
        data_provider=data_provider,
        sub_train_size=data_provider.subset_size,
        sub_train_batch_size=data_provider.subset_batch_size
    )

    if net_name == "OFAMobileNetV3":
        resume_ckpt = base_ckpt
    elif net_name == "SE_D_OFAMobileNetV3":
        resume_ckpt = SE_D_ckpt
    elif net_name == "EE_B_OFAMobileNetV3":
        resume_ckpt = EE_B_ckpt
    elif net_name == "EE_D_OFAMobileNetV3":
        resume_ckpt = EE_D_ckpt
    elif net_name == "SE_P_OFAMobileNetV3":
        resume_ckpt = SE_P_ckpt
    elif net_name == "SE_DP_OFAMobileNetV3":
        resume_ckpt = SE_DP_ckpt
    elif net_name == "EE_P_OFAMobileNetV3":
        resume_ckpt = EE_P_ckpt
    elif net_name == "EE_DP_OFAMobileNetV3":
        resume_ckpt = EE_DP_ckpt
    else:
        raise ValueError

    sd = torch.load(resume_ckpt, map_location='cpu')
    trainer_models_dicts = [sd["model_w1.0_state_dict"], sd["model_w1.2_state_dict"]]
    supernet.load_state_dict(trainer_models_dicts)
    supernet.cuda()

    return data_provider, search_space, supernet, evaluator


def get_attributes(archive, _attr):
    """
    :param archive: [{"acc": 53, "params": 123, "flops": 73, "latency": 62, "arch":{"r":64, "w":1.2, ... }},{...}]
    :param _attr: string with &-separated attributes to return
    :return: [[acc/err, params, flops, latency, arch], [acc/err, params, flops, latency, arch], ...]
    list of list of specified attributes (in _attr) for each element in the archive
    """

    attr_keys = _attr.split('&')
    batch_attr_values = []
    for member in archive:
        attr_values = []

        for attr_key in attr_keys:
            if attr_key == 'err':
                attr_values.append(100 - member['acc'])
            else:
                attr_values.append(member[attr_key])

        batch_attr_values.append(attr_values)
    return batch_attr_values


def fit_predictors(ss, surr, archive):
    data = get_attributes(archive, _attr='arch&err')  # [[dec_arch,err], [dec_arch,err], ...]

    # extract features from the encoded version of the architectures in the archive
    features = ss.features(ss.encode([d[0] for d in data]))
    # create array with corresponding target error values
    err_targets = np.array([d[1] for d in data])

    # obtain trained model to predict error given the encoded representation of a net
    err_predict = SurrogateModel(surr).fit(features, err_targets, ensemble=True)

    return err_predict


if __name__ == "__main__":

    seed = 420
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    bn_param = (0.1, 1e-5)
    dropout_rate = 0.1
    width_mult = 1.0
    ks_list = [3, 5, 7]
    expand_ratio_list = [3, 4, 6]
    depth_list = [2, 3, 4]
    net_depth_list = [1, 2, 3, 4, 5]
    net_width_list = [1, 2, 3]

    networks = [
        "OFAMobileNetV3",
        "SE_D_OFAMobileNetV3",
        "EE_B_OFAMobileNetV3",
        "EE_D_OFAMobileNetV3",  
        "SE_P_OFAMobileNetV3",
        "SE_DP_OFAMobileNetV3",
        "EE_P_OFAMobileNetV3",
        "EE_DP_OFAMobileNetV3",
    ]

    surrogates = ["rbf", "rbfs", "mlp", "e2epp", "carts", "gp", "svr", "ridge", "knn", "bayesian", "lgb", "catboost"]
    feature_encodings = ["one-hot", "integer"]
    n_train_samples = [300, 250, 200, 150, 100]
    n_test_samples = 250
    n_iter = 5

    parser = argparse.ArgumentParser()
    parser.add_argument("-n",choices=networks, required=True)
    args=parser.parse_args()
    net_name = args.n


    train_archives = {}
    test_archives = {}

    data_provider, search_space, supernet, evaluator = load_builders(net_name)
    
    test_archive_key = net_name
    test_samples = search_space.sample(n_test_samples)
    test_stats_dict = evaluator.evaluate(test_samples, objs="acc+params")
    test_archives[test_archive_key]={}
    test_archives[test_archive_key]["test_samples"] = test_samples
    test_archives[test_archive_key]["test_stats_dict"] = test_stats_dict

    
    for n_tr_samples in n_train_samples:
        for i in range(1, n_iter + 1):

            train_samples = search_space.sample(n_tr_samples)
            train_stats_dict = evaluator.evaluate(train_samples, objs="acc+params")
            train_archive = []
            for arch, stats in zip(train_samples, train_stats_dict):
                train_archive.append({'arch': arch, **stats})

            train_archive_key = f"{net_name}-{n_tr_samples}-{i}"
            train_archives[train_archive_key] = {}
            train_archives[train_archive_key]["train_archive"] = train_archive

    archives = {
        "train_archives": train_archives,
        "test_archives": test_archives
    }

    with open(f"running_experiment/pred_samples_{net_name}.json", "w") as f:
        json.dump(archives, f)

    results = {}

    for surrogate in surrogates:
        for feature_encoding in feature_encodings:
            data_provider, search_space, supernet, evaluator = load_builders(net_name, feature_encoding)
            for n_tr_samples in n_train_samples:
                for i in range(1, n_iter + 1):

                    if feature_encoding == "one-hot" and (surrogate == "rbf" or surrogate == "rbfs"):
                        pass
                    else:

                        train_archive_key = f"{net_name}-{n_tr_samples}-{i}"
                        test_archive_key = net_name

                        train_archive = train_archives[train_archive_key]["train_archive"]
                        test_samples = test_archives[test_archive_key]["test_samples"]
                        test_stats_dict = test_archives[test_archive_key]["test_stats_dict"]

                        start = time.time()

                        err_predictor = fit_predictors(search_space, surrogate, train_archive)
                        err_pred = SurrogateModel.predict(
                            model=err_predictor,
                            inputs=search_space.features(search_space.encode(test_samples))
                        )

                        if "rbf" in surrogate:
                            err_pred = err_pred[:, 0]

                        err_rmse, err_r, err_rho, err_tau = get_correlation(err_pred, [100 - stat['acc'] for stat in
                                                                                        test_stats_dict])
                        total_time_minutes = (time.time() - start) / 60

                        results_key = f"{surrogate}-{net_name}-{feature_encoding}-{n_tr_samples}"
                        if results_key not in results.keys():
                            results[results_key] = {}
                            results[results_key]["rmse"] = []
                            results[results_key]["r"] = []
                            results[results_key]["rho"] = []
                            results[results_key]["tau"] = []
                            results[results_key]["time_mins"] = []

                        results[results_key]["rmse"].append(err_rmse)
                        results[results_key]["r"].append(err_r)
                        results[results_key]["rho"].append(err_rho)
                        results[results_key]["tau"].append(err_tau)
                        results[results_key]["time_mins"].append(total_time_minutes)

                        with open(f"running_experiment/pred_comparisons_{net_name}.txt", "a") as f:
                            string = f"{surrogate}-{net_name}-{feature_encoding}-{n_tr_samples}-{i}\n"
                            string += f"err_rmse: {err_rmse:.6f} | err_r: {err_r:.6f} | err_rho: {err_rho:.6f} | err_tau: {err_tau:.6f}\n"
                            string += f"time: {total_time_minutes:.2f} mins\n\n"

                            f.write(string)

    with open(f"running_experiment/pred_results_{net_name}.json", "w") as f:
        json.dump(results, f)
