from utils import to_namedtuple

training_config = to_namedtuple('TrainingArg', {
    "base_path": ".",

    # todo: add dataset and update the <PascalVoc/>

    # model configs
    "epochs": 1,
    "batch_size_train": 8,
    "batch_size_test": 4,

    "is_colab": False,
    "path_colab_training": "/content/Imagenet",
    "path_local_training": "."

})

# imagenet_config // pre-training
# pascalvoc_config // object detection
#
