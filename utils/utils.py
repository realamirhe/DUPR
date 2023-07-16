from collections import namedtuple

import torchvision.transforms.functional as torchvisionF
from torch import nn


def lens(dic, path):
    res = dic
    for i in path.split('.'):
        res = res[int(i) if i.isdigit() else i]
    return res


class PILToTensor(nn.Module):
    def forward(self, image):
        image = torchvisionF.pil_to_tensor(image)
        image = torchvisionF.convert_image_dtype(image)
        return image


def collate_fn(batch):
    return tuple(zip(*batch))


def to_namedtuple(name, dictionary):
    return namedtuple(name, dictionary)(**dictionary)


def clear_colab_gpu_ram():
    # TODO: must be hoisted and installed in the colab
    # !pip install numba - q
    from numba import cuda

    cuda.select_device(0)
    cuda.close()

    device = cuda.get_current_device()
    device.reset()
