import os
import warnings

import PIL.Image as Image
import numpy as np
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from ofa.imagenet_classification.data_providers.base_provider import DataProvider
from ofa.utils import MyDistributedSampler

from OFA_mbv3_extended.my_data_loaders.my_random_resize_crop import MyModRandomResizedCrop


class Flowers102DataProvider(DataProvider):
    
    def __init__(
        self,
        save_path=None,
        train_batch_size=256,
        test_batch_size=512,
        valid_size=1000,
        n_worker=8,
        resize_scale=1.0,
        distort_color=None,
        image_size=64,
        num_replicas=None,
        rank=None,
        subset_size=150,
        subset_batch_size=100
    ):
        # only non-distributed

        warnings.filterwarnings('ignore')
        self._save_path = save_path
        
        self.image_size = image_size  # int or list of int
        self.distort_color = distort_color
        self.resize_scale = resize_scale

        self.subset_size = subset_size
        self.subset_batch_size = subset_batch_size

        self.valid_size = valid_size

        self._valid_transform_dict = {}
        if not isinstance(self.image_size, int):
            assert isinstance(self.image_size, list)
            from OFA_mbv3_extended.my_data_loaders.my_data_loader import MyDataLoader
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

        weights = self.make_weights_for_balanced_classes(train_dataset.imgs, self.n_classes)
        weights = torch.DoubleTensor(weights)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        self.train = train_loader_class(
            train_dataset,
            batch_size=train_batch_size,
            sampler=train_sampler,
            num_workers=n_worker,
            pin_memory=True,
        )

        test_dataset = self.test_dataset(valid_transforms)

        if valid_size is not None:
            if not isinstance(valid_size, int):
                assert isinstance(valid_size, float) and 0 < valid_size < 1
                valid_size = int(len(test_dataset.samples) * valid_size)

            # we use a partial testing set for validation
            valid_dataset = self.test_dataset(valid_transforms)
            test_indexes, valid_indexes = self.random_sample_valid_set(len(test_dataset.samples), valid_size)

            test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indexes)
            valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)

            self.test = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                sampler=test_sampler,
                num_workers=n_worker,
                pin_memory=True,
            )

            self.valid = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=test_batch_size,
                sampler=valid_sampler,
                num_workers=n_worker,
                pin_memory=True,
            )

        else:
            raise ValueError("validation set size must be specified for this dataset")
    
    @staticmethod
    def name():
        return 'flowers102'
    
    @property
    def data_shape(self):
        return 3, self.active_img_size, self.active_img_size  # C, H, W
    
    @property
    def n_classes(self):
        return 102
    
    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = 'running_experiment/datasets/private/Flowers102'

            if not os.path.exists(self._save_path):
                self._save_path = 'running_experiment/datasets/private/Flowers102'
        return self._save_path
    
    @property
    def data_url(self):
        raise ValueError('unable to download %s' % self.name())
    
    def train_dataset(self, _transforms):
        dataset = datasets.ImageFolder(self.train_path, _transforms)
        # dataset = torchvision.datasets.Flowers102(root=self.train_path, split="train", download=True, transform=_transforms)
        return dataset

    def valid_dataset(self, _transforms):
        raise NotImplementedError('no validation dataset is available')
        # dataset = torchvision.datasets.Flowers102(root=self.valid_path, split="val", download=True, transform=_transforms)
        # return dataset

    def test_dataset(self, _transforms):
        dataset = datasets.ImageFolder(self.test_path, _transforms)
        # dataset = torchvision.datasets.Flowers102(root=self.test_path, split="test", download=True, transform=_transforms)
        return dataset
    
    @property
    def train_path(self):
        return os.path.join(self.save_path, 'train')
        # return self.save_path

    @property
    def valid_path(self):
        raise ValueError('no validation data is available')
        # return self.save_path

    @property
    def test_path(self):
        return os.path.join(self.save_path, 'test')
        # return self.save_path
    
    @property
    def normalize(self):
        return transforms.Normalize(
            mean=[0.5178361839861569, 0.4106749456881299, 0.32864167836880803],
            std=[0.2972239085211309, 0.24976049135203868, 0.28533308036347665]
        )

    @staticmethod
    def make_weights_for_balanced_classes(images, nclasses):
        count = [0] * nclasses

        # Counts per label
        for item in images:
            count[item[1]] += 1

        weight_per_class = [0.] * nclasses

        # Total number of images.
        num_images = float(sum(count))

        # super-sample the smaller classes.
        for i in range(nclasses):
            weight_per_class[i] = num_images / float(count[i])

        weight = [0] * len(images)

        # Calculate a weight per image.
        for idx, val in enumerate(images):
            weight[idx] = weight_per_class[val[1]]

        return weight

    def build_train_transform(self, image_size=None, print_log=True):
        if image_size is None:
            image_size = self.image_size
        if print_log:
            print('Color jitter: %s, valid_set_size: %s, resize_scale: %s, img_size: %s' %
                  (self.distort_color, self.valid_size, self.resize_scale, image_size))
        
        if isinstance(image_size, list):
            resize_transform_class = MyModRandomResizedCrop
            print('Use MyModRandomResizedCrop: %s, \t %s'
                  % MyModRandomResizedCrop.get_candidate_image_size(),
                  'sync=%s, continuous=%s' %
                  (MyModRandomResizedCrop.SYNC_DISTRIBUTED,
                   MyModRandomResizedCrop.CONTINUOUS))
        else:
            resize_transform_class = transforms.RandomResizedCrop

        train_transforms = [
            transforms.RandomAffine(
                degrees=45,
                translate=(0.4, 0.4),
                scale=(0.75, 1.5),
                shear=None,
                resample=Image.BILINEAR,
                fillcolor=0
            ),
            resize_transform_class(image_size, scale=(self.resize_scale, 1.0)),
            transforms.ToTensor(),
            self.normalize,
        ]

        train_transforms = transforms.Compose(train_transforms)
        return train_transforms

    def build_valid_transform(self, image_size=None):
        if image_size is None:
            image_size = self.active_img_size
        return transforms.Compose([
            # transforms.Resize(int(math.ceil(image_size / 0.875))),
            # transforms.CenterCrop(image_size),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            self.normalize,
        ])

    def assign_active_img_size(self, new_img_size):
        self.active_img_size = new_img_size
        if self.active_img_size not in self._valid_transform_dict:
            self._valid_transform_dict[self.active_img_size] = self.build_valid_transform()
        # change the transform of the valid and test set
        try:
            self.valid.dataset.transform = self._valid_transform_dict[self.active_img_size]
        except AttributeError:
            pass
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

