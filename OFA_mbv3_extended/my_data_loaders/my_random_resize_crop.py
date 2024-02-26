import os
import random
import time

import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode

__all__ = ["MyModRandomResizedCrop"]

_pil_interpolation_to_str = {
    InterpolationMode.NEAREST: "InterpolationMode.NEAREST",
    InterpolationMode.BILINEAR: "InterpolationMode.BILINEAR",
    InterpolationMode.BICUBIC: "InterpolationMode.BICUBIC",
    InterpolationMode.LANCZOS: "InterpolationMode.LANCZOS",
    InterpolationMode.HAMMING: "InterpolationMode.HAMMING",
    InterpolationMode.BOX: "InterpolationMode.BOX",
}


class MyModRandomResizedCrop(transforms.RandomResizedCrop):
    ACTIVE_SIZE = 224
    IMAGE_SIZE_LIST = [224]
    IMAGE_SIZE_SEG = 4

    CONTINUOUS = False
    SYNC_DISTRIBUTED = True

    EPOCH = 0
    BATCH = 0

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=InterpolationMode.BILINEAR,
    ):

        if not isinstance(size, int):
            size = size[0]
        super(MyModRandomResizedCrop, self).__init__(size, scale, ratio, interpolation)

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(
            img,
            i,
            j,
            h,
            w,
            [MyModRandomResizedCrop.ACTIVE_SIZE, MyModRandomResizedCrop.ACTIVE_SIZE],
            self.interpolation,
        )

    @staticmethod
    def get_candidate_image_size():
        if MyModRandomResizedCrop.CONTINUOUS:
            min_size = min(MyModRandomResizedCrop.IMAGE_SIZE_LIST)
            max_size = max(MyModRandomResizedCrop.IMAGE_SIZE_LIST)
            candidate_sizes = []
            for i in range(min_size, max_size + 1):
                if i % MyModRandomResizedCrop.IMAGE_SIZE_SEG == 0:
                    candidate_sizes.append(i)
        else:
            candidate_sizes = MyModRandomResizedCrop.IMAGE_SIZE_LIST

        relative_probs = None
        return candidate_sizes, relative_probs

    @staticmethod
    def sample_image_size(batch_id=None):
        if batch_id is None:
            batch_id = MyModRandomResizedCrop.BATCH
        if MyModRandomResizedCrop.SYNC_DISTRIBUTED:
            _seed = int("%d%.3d" % (batch_id, MyModRandomResizedCrop.EPOCH))
        else:
            _seed = os.getpid() + time.time()
        random.seed(_seed)
        candidate_sizes, relative_probs = MyModRandomResizedCrop.get_candidate_image_size()
        MyModRandomResizedCrop.ACTIVE_SIZE = random.choices(
            candidate_sizes, weights=relative_probs
        )[0]

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + "(size={0}".format(
            MyModRandomResizedCrop.IMAGE_SIZE_LIST
        )
        if MyModRandomResizedCrop.CONTINUOUS:
            format_string += "@continuous"
        format_string += ", scale={0}".format(tuple(round(s, 4) for s in self.scale))
        format_string += ", ratio={0}".format(tuple(round(r, 4) for r in self.ratio))
        format_string += ", interpolation={0})".format(interpolate_str)
        return format_string