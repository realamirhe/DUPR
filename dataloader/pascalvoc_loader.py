from operator import itemgetter

import torch
from torch.utils.data import ConcatDataset
from torchvision import transforms
from torchvision.datasets import VOCDetection

from utils.utils import lens, PILToTensor


class PascalVoc:
    CLASSES_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                     'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                     'tvmonitor', 'ambigious']
    CLASSES_2_LABEL = {k: i for i, k in enumerate(CLASSES_NAMES)}

    def __init__(self, mode='train'):
        is_train_mode = mode == 'train'
        image_set = "trainval" if is_train_mode else "test"

        shared_config = {
            "root": "./pascal-voc",
            "target_transform": self.target_transform,
            "transform": transforms.Compose([PILToTensor()]),
            "download": True
        }

        if is_train_mode:
            self.data = ConcatDataset([
                VOCDetection(image_set=image_set, year="2007", **shared_config),
                VOCDetection(image_set=image_set, year="2012", **shared_config)
            ])
        else:
            self.data = VOCDetection(image_set=image_set, year="2007", **shared_config),

    @staticmethod
    def target_transform(raw_target):
        targets = []
        for obj in lens(raw_target, 'annotation.object'):
            xmin, ymin, xmax, ymax = itemgetter('xmin', 'ymin', 'xmax', 'ymax')(lens(obj, 'bndbox'))
            targets.append({
                "labels": torch.tensor(PascalVoc.CLASSES_2_LABEL[lens(obj, 'name')]),
                "boxes": torch.tensor(list(map(int, [xmin, ymin, xmax, ymax]))).unsqueeze(0)
            })
        return targets
