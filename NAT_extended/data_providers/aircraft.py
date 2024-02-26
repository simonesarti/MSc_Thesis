from __future__ import print_function

import math
import os
import warnings

import numpy as np
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from ofa.imagenet_classification.data_providers.base_provider import DataProvider
from ofa.utils.my_dataloader import MyDistributedSampler
from timm.data.auto_augment import rand_augment_transform

from OFA_mbv3_extended.my_data_loaders.my_data_loader import MyDataLoader
from OFA_mbv3_extended.my_data_loaders.my_random_resize_crop import MyModRandomResizedCrop


class FGVCAircraftDataProvider(DataProvider):

    def __init__(
        self,
        save_path=None,
        train_batch_size=256,
        test_batch_size=512,
        valid_size=None,
        n_worker=8,
        resize_scale=0.35,
        distort_color=None,
        image_size=64,
        num_replicas=None,
        rank=None,
        subset_size=200,
        subset_batch_size=100
    ):

        warnings.filterwarnings('ignore')
        self._save_path = save_path

        self.image_size = image_size  # int or list of int
        self.distort_color = distort_color
        self.resize_scale = resize_scale

        self.subset_size = subset_size
        self.subset_batch_size = subset_batch_size

        self._valid_transform_dict = {}
        if not isinstance(self.image_size, int):
            assert isinstance(self.image_size, list)

            self.image_size.sort()
            MyModRandomResizedCrop.IMAGE_SIZE_LIST = self.image_size.copy()
            MyModRandomResizedCrop.ACTIVE_SIZE = max(self.image_size)

            for img_size in self.image_size:
                self._valid_transform_dict[img_size] = self.build_valid_transform(img_size)
            self.active_img_size = max(self.image_size)
            valid_transforms = self._valid_transform_dict[self.active_img_size]
            train_loader_class = MyDataLoader  # randomly sample image size for each batch of training image
        else:
            self.active_img_size = self.image_size
            valid_transforms = self.build_valid_transform()
            train_loader_class = torch.utils.data.DataLoader

        train_transforms = self.build_train_transform()
        train_dataset = self.train_dataset(train_transforms)

        # validation set is present for the dataset, use it
        valid_dataset = self.valid_dataset(valid_transforms)

        if num_replicas is not None:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas, rank)
            self.train = train_loader_class(
                train_dataset,
                batch_size=train_batch_size,
                sampler=train_sampler,
                num_workers=n_worker,
                pin_memory=True
            )

            valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, num_replicas, rank)
            self.valid = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=test_batch_size,
                sampler=valid_sampler,
                num_workers=n_worker,
                pin_memory=True,
            )
        else:
            self.train = train_loader_class(
                train_dataset,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=n_worker,
                pin_memory=True,
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=test_batch_size,
                num_workers=n_worker,
                pin_memory=True,
            )

        test_dataset = self.test_dataset(valid_transforms)
        if num_replicas is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas, rank)
            self.test = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                sampler=test_sampler,
                num_workers=n_worker,
                pin_memory=True,
            )
        else:
            self.test = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                num_workers=n_worker,
                pin_memory=True,
            )

    @staticmethod
    def name():
        return 'aircraft'

    @property
    def data_shape(self):
        return 3, self.active_img_size, self.active_img_size  # C, H, W

    @property
    def n_classes(self):
        return 100

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = 'running_experiment/datasets/private/Aircraft'

            if not os.path.exists(self._save_path):
                self._save_path = 'running_experiment/datasets/private/Aircraft'
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download %s' % self.name())

    def train_dataset(self, _transforms):
        dataset = torchvision.datasets.FGVCAircraft(
            root=self.train_path,
            split="train",
            annotation_level="variant",
            download=True,
            transform=_transforms
        )
        return dataset

    def valid_dataset(self, _transforms):
        dataset = torchvision.datasets.FGVCAircraft(
                root=self.valid_path,
                split="val",
                annotation_level="variant",
                download=True,
                transform=_transforms
            )
        return dataset

    def test_dataset(self, _transforms):
        dataset = torchvision.datasets.FGVCAircraft(
                root=self.test_path,
                split="test",
                annotation_level="variant",
                download=True,
                transform=_transforms
            )
        return dataset

    @property
    def train_path(self):
        return self.save_path

    @property
    def valid_path(self):
        return self.save_path

    @property
    def test_path(self):
        return self.save_path

    @property
    def normalize(self):
        return transforms.Normalize(
            mean=[0.48933587508932375, 0.5183537408957618, 0.5387914411673883],
            std=[0.22388883112804625, 0.21641635409388751, 0.24615605842636115]
        )

    def build_train_transform(self, image_size=None, print_log=True, auto_augment='rand-m9-mstd0.5'):
        if image_size is None:
            image_size = self.image_size
        if print_log:
            print('Color jitter: %s, resize_scale: %s, img_size: %s' %
                  (self.distort_color, self.resize_scale, image_size))

        # if self.distort_color == 'torch':
        #     color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        # elif self.distort_color == 'tf':
        #     color_transform = transforms.ColorJitter(brightness=32. / 255., saturation=0.5)
        # else:
        #     color_transform = None

        if isinstance(image_size, list):
            resize_transform_class = MyModRandomResizedCrop
            print('Use MyModRandomResizedCrop: %s, \t %s'
                  % MyModRandomResizedCrop.get_candidate_image_size(),
                  'sync=%s, continuous=%s' % (
                      MyModRandomResizedCrop.SYNC_DISTRIBUTED,
                      MyModRandomResizedCrop.CONTINUOUS))
            img_size_min = min(image_size)
        else:
            resize_transform_class = transforms.RandomResizedCrop
            img_size_min = image_size

        train_transforms = [
            resize_transform_class(image_size, scale=(self.resize_scale, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]

        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in [0.48933587508932375, 0.5183537408957618,
                                                               0.5387914411673883]]),
        )
        aa_params['interpolation'] = Image.BICUBIC
        train_transforms += [rand_augment_transform(auto_augment, aa_params)]

        # if color_transform is not None:
        #     train_transforms.append(color_transform)
        train_transforms += [
            transforms.ToTensor(),
            self.normalize,
        ]

        train_transforms = transforms.Compose(train_transforms)
        return train_transforms

    def build_valid_transform(self, image_size=None):
        if image_size is None:
            image_size = self.active_img_size
        return transforms.Compose([
            transforms.Resize(int(math.ceil(image_size / 0.875))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            self.normalize,
        ])

    def assign_active_img_size(self, new_img_size):
        self.active_img_size = new_img_size
        if self.active_img_size not in self._valid_transform_dict:
            self._valid_transform_dict[self.active_img_size] = self.build_valid_transform()
        # change the transform of the valid and test set
        self.valid.dataset.transform = self._valid_transform_dict[self.active_img_size]
        self.test.dataset.transform = self._valid_transform_dict[self.active_img_size]

    def build_sub_train_loader(self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None):
        # used for resetting running statistics
        if self.__dict__.get('sub_train_%d' % self.active_img_size, None) is None:
            if num_worker is None:
                num_worker = self.train.num_workers

            n_samples = len(self.train.dataset)
            g = torch.Generator()
            g.manual_seed(DataProvider.SUB_SEED)
            rand_indexes = torch.randperm(n_samples, generator=g).tolist()

            new_train_dataset = self.train_dataset(
                self.build_train_transform(image_size=self.active_img_size, print_log=False))

            chosen_indexes = rand_indexes[:n_images]
            if num_replicas is not None:
                sub_sampler = MyDistributedSampler(new_train_dataset, num_replicas, rank, True, np.array(chosen_indexes))
            else:
                sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)

            sub_data_loader = torch.utils.data.DataLoader(
                new_train_dataset,
                batch_size=batch_size,
                sampler=sub_sampler,
                num_workers=num_worker,
                pin_memory=True,
            )
            self.__dict__['sub_train_%d' % self.active_img_size] = []
            for images, labels in sub_data_loader:
                self.__dict__['sub_train_%d' % self.active_img_size].append((images, labels))
        return self.__dict__['sub_train_%d' % self.active_img_size]

