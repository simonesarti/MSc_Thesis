from ofa.imagenet_classification.run_manager import RunConfig

from OFA_mbv3_extended.data_providers.imagenet import ImagenetDataProvider
from OFA_mbv3_extended.data_providers.tiny_imagenet import TinyImagenetDataProvider

__all__ = [
   "DatasetRunConfig",
   "DistributedDatasetRunConfig"
]


class DatasetRunConfig(RunConfig):

    def __init__(
        self,
        dataset,
        n_epochs=150,
        init_lr=0.05,
        lr_schedule_type="cosine",
        lr_schedule_param=None,
        train_batch_size=256,
        test_batch_size=512,
        valid_size=None,
        opt_type="sgd",
        opt_param=None,
        weight_decay=4e-5,
        label_smoothing=0.1,
        no_decay_keys=None,
        mixup_alpha=None,
        model_init="he_fout",
        validation_frequency=1,
        print_frequency=10,
        n_worker=8,
        resize_scale=1.0,
        distort_color='tf',
        image_size=64,
        branches_weights=None,
        ensemble_weights=None,
        patience=8,
        n_elastic_val=0,
        **kwargs
    ):
        super(DatasetRunConfig, self).__init__(
            n_epochs,
            init_lr,
            lr_schedule_type,
            lr_schedule_param,
            dataset,
            train_batch_size,
            test_batch_size,
            valid_size,
            opt_type,
            opt_param,
            weight_decay,
            label_smoothing,
            no_decay_keys,
            mixup_alpha,
            model_init,
            validation_frequency,
            print_frequency,
        )
        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color
        self.image_size = image_size
        self.branches_weights = branches_weights
        self.ensemble_weights = ensemble_weights
        self.patience = patience
        self.n_elastic_val = n_elastic_val

    @property
    def data_provider(self):
        if self.__dict__.get("_data_provider", None) is None:
            if self.dataset == "tiny_imagenet":
                DataProviderClass = TinyImagenetDataProvider
            elif self.dataset == "imagenet":
                DataProviderClass = ImagenetDataProvider
            else:
                raise NotImplementedError
            self.__dict__["_data_provider"] = DataProviderClass(
                train_batch_size=self.train_batch_size,
                test_batch_size=self.test_batch_size,
                valid_size=self.valid_size,
                n_worker=self.n_worker,
                resize_scale=self.resize_scale,
                distort_color=self.distort_color,
                image_size=self.image_size,
            )
        return self.__dict__["_data_provider"]


class DistributedDatasetRunConfig(DatasetRunConfig):

    def __init__(
        self,
        dataset,
        n_epochs=150,
        init_lr=0.05,
        lr_schedule_type="cosine",
        lr_schedule_param=None,
        train_batch_size=64,
        test_batch_size=64,
        valid_size=None,
        opt_type="sgd",
        opt_param=None,
        weight_decay=4e-5,
        label_smoothing=0.1,
        no_decay_keys=None,
        mixup_alpha=None,
        model_init="he_fout",
        validation_frequency=1,
        print_frequency=10,
        n_worker=8,
        resize_scale=1.0,
        distort_color=None,
        image_size=64,
        branches_weights=None,
        ensemble_weights=None,
        patience=8,
        n_elastic_val=0,
        **kwargs
    ):
        super(DistributedDatasetRunConfig, self).__init__(
            dataset,
            n_epochs,
            init_lr,
            lr_schedule_type,
            lr_schedule_param,
            train_batch_size,
            test_batch_size,
            valid_size,
            opt_type,
            opt_param,
            weight_decay,
            label_smoothing,
            no_decay_keys,
            mixup_alpha,
            model_init,
            validation_frequency,
            print_frequency,
            n_worker,
            resize_scale,
            distort_color,
            image_size,
            branches_weights,
            ensemble_weights,
            patience,
            n_elastic_val,
            **kwargs
        )
        self._num_replicas = kwargs["num_replicas"]
        self._rank = kwargs["rank"]

    @property
    def data_provider(self):
        if self.__dict__.get("_data_provider", None) is None:
            if self.dataset == "tiny_imagenet":
                DataProviderClass = TinyImagenetDataProvider
            elif self.dataset == "imagenet":
                DataProviderClass = ImagenetDataProvider
            else:
                raise NotImplementedError
            self.__dict__["_data_provider"] = DataProviderClass(
                train_batch_size=self.train_batch_size,
                test_batch_size=self.test_batch_size,
                valid_size=self.valid_size,
                n_worker=self.n_worker,
                resize_scale=self.resize_scale,
                distort_color=self.distort_color,
                image_size=self.image_size,
                num_replicas=self._num_replicas,
                rank=self._rank,
            )
        return self.__dict__["_data_provider"]




