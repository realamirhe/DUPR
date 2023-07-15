import glob
import os
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


def load_files(train_path, val_path):
    try:
        from google.colab import drive
        drive.mount('/content/GDrive/')
        path = "/content/imagenette2-320"
    except:
        # TODO: add the local load stage!
        raise Exception('please run the code on the colab')
    train_path = os.path.join(path, train_path)
    val_path = os.path.join(path, val_path)
    classes = os.listdir(train_path)
    print("Total Classes: ", len(classes))

    train_imgs = []
    val_imgs = []
    for _class in classes:
        train_imgs += glob.glob(os.path.join(train_path, _class) + '/*', recursive=True)
        val_imgs += glob.glob(os.path.join(val_path, _class) + '/*', recursive=True)

    print("Total train images: ", len(train_imgs))
    print("Total test images: ", len(val_imgs))

    return train_imgs, val_imgs


def clear_colab_gpu_ram():
    # TODO: must be hoisted and installed in the colab
    # !pip install numba - q
    from numba import cuda

    cuda.select_device(0)
    cuda.close()

    device = cuda.get_current_device()
    device.reset()