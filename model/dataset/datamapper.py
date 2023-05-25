import copy
import logging

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Boxes, Instances

__all__ = ["COCOPanopticNewBaselineDatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])

    return augmentation

class COCOPanopticNewBaselineDatasetMapper:
    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
    ):
        self.is_train = is_train
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[COCOPanopticNewBaselineDatasetMapper] Full TransformGens used in training: {}".format(
                str(self.tfm_gens)
            )
        )
        
        self.image_format = image_format
        
    def from_config(cls, cfg, is_train=True):
        #build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)
        
        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
        }
        
        return ret
    
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        #read image
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        
        #augmentation
        input = T.AugInput(image)
        image, transforms = T.apply_augmentations(self.tfm_gens, input)
        image_shape = image[:2] #h, w
        
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        
        return dataset_dict