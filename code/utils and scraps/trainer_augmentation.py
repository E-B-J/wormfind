from detectron2.engine import DefaultTrainer
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper, build_detection_train_loader

import numpy as np

class RandomScale(T.Augmentation):
    def __init__(
        self,
        min_scale: float,
        max_scale: float,):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale

    def get_transform(self, image):
        old_h, old_w = image.shape[:2]
        sacle = np.random.uniform(self.min_scale, self.max_scale)
        new_h, new_w = int(old_h * sacle), int(old_w * sacle)
        return T.ResizeTransform(old_h, old_w, new_h, new_w)

class BuchneraTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=[
                                                                                            T.RandomBrightness(0.5, 2),
                                                                                            T.RandomContrast(0.5, 2),
                                                                                            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                                                                                            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                                                                                            RandomScale(0.5,1.5)]))