import os


def assign_phases_args(args):

    if args.task == "full":
        args.dynamic_batch_size = 0
        args.n_epochs = 1#180
        args.base_lr = 0.08
        args.warmup_epochs = 0
        args.warmup_lr = -1
        args.ks_list = "7"
        args.net_width_list = "3"
        args.net_depth_list = "5"
        args.expand_list = "6"
        args.depth_list = "4"

    elif args.task == "kernel":
        args.dynamic_batch_size = 1
        args.n_epochs = 1#120
        args.base_lr = 0.03
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.ks_list = "3,5,7"
        args.net_width_list = "3"
        args.net_depth_list = "5"
        args.expand_list = "6"
        args.depth_list = "4"
    ################################################################################
    elif args.task == "net_width":
        # only parallel networks support this operation
        if "P" not in args.network:
            raise ValueError("The network doesn't support this operation")
        else:
            args.dynamic_batch_size = 2
            if args.phase == 1:
                args.n_epochs = 25
                args.base_lr = 2.5e-3
                args.warmup_epochs = 0
                args.warmup_lr = -1
                args.ks_list = "3,5,7"
                args.net_width_list = "2,3"
                args.net_depth_list = "5"
                args.depth_list = "4"
                args.expand_list = "6"
            elif args.phase == 2:
                args.n_epochs = 120
                args.base_lr = 7.5e-3
                args.warmup_epochs = 5
                args.warmup_lr = -1
                args.ks_list = "3,5,7"
                args.net_width_list = "1,2,3"
                args.net_depth_list = "5"
                args.depth_list = "4"
                args.expand_list = "6"
            else:
                raise ValueError("net-width mode only supports phases [1,2]")
    ################################################################################
    elif args.task == "net_depth":
        if "EE" not in args.network:
            raise ValueError("Single exit networks do not support elastic net-depth")
        else:
            args.dynamic_batch_size = 2
            if args.phase == 1:
                args.n_epochs = 1#25
                args.base_lr = 0.001875
                args.warmup_epochs = 0
                args.warmup_lr = -1
                args.ks_list = "3,5,7"
                args.net_width_list = "1,2,3"
                args.net_depth_list = "4,5"
                args.depth_list = "4"
                args.expand_list = "6"
            elif args.phase == 2:
                args.n_epochs = 1#60
                args.base_lr = 0.00375
                args.warmup_epochs = 5
                args.warmup_lr = -1
                args.ks_list = "3,5,7"
                args.net_width_list = "1,2,3"
                args.net_depth_list = "3,4,5"
                args.depth_list = "4"
                args.expand_list = "6"
            elif args.phase == 3:
                args.n_epochs = 1#90
                args.base_lr = 0.0075
                args.warmup_epochs = 5
                args.warmup_lr = -1
                args.ks_list = "3,5,7"
                args.net_width_list = "1,2,3"
                args.net_depth_list = "2,3,4,5"
                args.depth_list = "4"
                args.expand_list = "6"
            elif args.phase == 4:
                args.n_epochs = 1#120
                args.base_lr = 0.015
                args.warmup_epochs = 5
                args.warmup_lr = -1
                args.ks_list = "3,5,7"
                args.net_width_list = "1,2,3"
                args.net_depth_list = "1,2,3,4,5"
                args.depth_list = "4"
                args.expand_list = "6"
            else:
                raise ValueError("net-depth mode only supports phases [1,2,3,4]")

    elif args.task == "depth":
        args.dynamic_batch_size = 2
        if args.phase == 1:
            args.n_epochs = 1#25
            args.base_lr = 0.0025
            args.warmup_epochs = 0
            args.warmup_lr = -1
            args.ks_list = "3,5,7"
            args.net_width_list = "1,2,3"
            args.net_depth_list = "1,2,3,4,5"
            args.depth_list = "3,4"
            args.expand_list = "6"
        elif args.phase == 2:
            args.n_epochs = 1#120
            args.base_lr = 0.0075
            args.warmup_epochs = 5
            args.warmup_lr = -1
            args.ks_list = "3,5,7"
            args.net_width_list = "1,2,3"
            args.net_depth_list = "1,2,3,4,5"
            args.depth_list = "2,3,4"
            args.expand_list = "6"
        else:
            raise ValueError("depth mode only supports phases [1,2]")

    elif args.task == "expand":
        args.dynamic_batch_size = 4
        if args.phase == 1:
            args.n_epochs = 1#25
            args.base_lr = 0.0025
            args.warmup_epochs = 0
            args.warmup_lr = -1
            args.ks_list = "3,5,7"
            args.net_width_list = "1,2,3"
            args.net_depth_list = "1,2,3,4,5"
            args.depth_list = "2,3,4"
            args.expand_list = "4,6"
        elif args.phase == 2:
            args.n_epochs = 1#120
            args.base_lr = 0.0075
            args.warmup_epochs = 5
            args.warmup_lr = -1
            args.ks_list = "3,5,7"
            args.net_width_list = "1,2,3"
            args.net_depth_list = "1,2,3,4,5"
            args.depth_list = "2,3,4"
            args.expand_list = "3,4,6"
        else:
            raise ValueError("expand mode only supports phases [1,2]")

    else:
        raise NotImplementedError


def get_paths_dict(args):
    paths = {}
    init = os.path.join(
        "running_experiment",
        "storage",
        "OFA",
        "train",
        args.dataset,
        args.network,
        "wm_" + str(args.width_mult),
        "Nexp_" + str(args.n_exp)
    )

    if args.task in ["full", "kernel"]:
        paths["current"] = os.path.join(init, args.task)
    else:
        paths["current"] = os.path.join(init, args.task, "phase_" + str(args.phase))

    best_chkpt = os.path.join("checkpoint", "model_best.pth.tar")

    paths["teacher"] = os.path.join(init, "full", best_chkpt)
    paths["kernel"] = os.path.join(init, "full", best_chkpt)

    if "P" in args.network and "EE" in args.network:
        paths["net_width_1"] = os.path.join(init, "kernel", best_chkpt)
        paths["net_width_2"] = os.path.join(init, "net_width", "phase_1", best_chkpt)
        paths["net_depth_1"] = os.path.join(init, "net_width", "phase_2", best_chkpt)
        paths["net_depth_2"] = os.path.join(init, "net_depth", "phase_1", best_chkpt)
        paths["net_depth_3"] = os.path.join(init, "net_depth", "phase_2", best_chkpt)
        paths["net_depth_4"] = os.path.join(init, "net_depth", "phase_3", best_chkpt)
        paths["depth_1"] = os.path.join(init, "net_depth", "phase_4", best_chkpt)

    elif "P" in args.network and "EE" not in args.network:
        paths["net_width_1"] = os.path.join(init, "kernel", best_chkpt)
        paths["net_width_2"] = os.path.join(init, "net_width", "phase_1", best_chkpt)
        paths["depth_1"] = os.path.join(init, "net_width", "phase_2", best_chkpt)

    elif "P" not in args.network and "EE" in args.network:
        paths["net_depth_1"] = os.path.join(init, "kernel", best_chkpt)
        paths["net_depth_2"] = os.path.join(init, "net_depth", "phase_1", best_chkpt)
        paths["net_depth_3"] = os.path.join(init, "net_depth", "phase_2", best_chkpt)
        paths["net_depth_4"] = os.path.join(init, "net_depth", "phase_3", best_chkpt)
        paths["depth_1"] = os.path.join(init, "net_depth", "phase_4", best_chkpt)

    else:
        paths["depth_1"] = os.path.join(init, "kernel", best_chkpt)

    paths["depth_2"] = os.path.join(init, "depth", "phase_1", best_chkpt)
    paths["expand_1"] = os.path.join(init, "depth", "phase_2", best_chkpt)
    paths["expand_2"] = os.path.join(init, "expand", "phase_1", best_chkpt)

    return paths


def get_previous_path(args, paths_dict):

    if args.task == "kernel":
        return paths_dict["teacher"]

    if args.phase > 1:
        previous_key = f"{args.task}_{args.phase-1}"
        return paths_dict[previous_key]

    if args.task == "expand":
        return paths_dict["depth_2"]

    if "P" in args.network and "EE" in args.network:
        if args.task == "depth":
            return paths_dict["net_depth_4"]
        if args.task == "net_depth":
            return paths_dict["net_width_2"]
        if args.task == "net_width":
            return paths_dict["kernel"]

    if "P" in args.network and "EE" not in args.network:
        if args.task == "depth":
            return paths_dict["net_width_2"]
        if args.task == "net_width":
            return paths_dict["kernel"]

    if "P" not in args.network and "EE" in args.network:
        if args.task == "depth":
            return paths_dict["net_depth_4"]
        if args.task == "net_depth":
            return paths_dict["kernel"]

    if "P" not in args.network and "EE" not in args.network:
        if args.task == "depth":
            return paths_dict["kernel"]

def assign_run_args(args):

    args.manual_seed = 0

    args.base_batch_size = 64

    args.opt_type = "sgd"
    args.lr_schedule_type = "cosine"
    args.momentum = 0.9
    args.no_nesterov = False
    args.weight_decay = 3e-5
    args.label_smoothing = 0.1
    args.no_decay_keys = "bn#bias"
    args.fp16_allreduce = False

    args.model_init = "he_fout"
    args.validation_frequency = 1
    args.print_frequency = 10

    args.n_worker = 0
    args.distort_color = None

    args.continuous_size = True
    args.not_sync_distributed_image_size = False

    args.bn_momentum = 0.1
    args.bn_eps = 1e-5
    args.dropout = 0.1

    args.dy_conv_scaling_mode = 1
    args.independent_distributed_sampling = False

    args.kd_ratio = 1.0
    args.kd_type = "ce"

    args.branches_weights = [0.06, 0.13, 0.20, 0.27, 0.34]
    args.ensemble_weights = [0.06, 0.13, 0.20, 0.27, 0.34]
    args.patience = 8
    args.n_elastic_val = 32


def assign_dataset_args(args):

    if args.dataset == "imagenet":
        args.valid_size = 10000
        args.image_size = "128,160,192,224"
        args.resize_scale = 0.08

    elif args.dataset == "tiny_imagenet":
        args.valid_size = 0.15
        args.image_size = "48,56,64"
        args.resize_scale = 0.85
    else:
        raise NotImplementedError



