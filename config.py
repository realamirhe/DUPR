from utils import to_namedtuple

pretraining_conf = to_namedtuple('PreTrainingConfig', {
    # Configuration
    "version": "0.0.1",

    "base_path": ".",
    # model configs
    "epochs": 25,
    "save_freq": 5,
    "batch_size_DUPR": 8,
    "batch_size_pascalVOC_train": 8,
    "batch_size_pascalVOC_test": 4,

    "optimizer": to_namedtuple('Optimizer', {
        "lr": 0.0001,
        "gamma": 0.5,
        "step": 1
    }),

    # Latest checkpoint to load
    "checkpoint": "ckpt_DUPR_epoch20.ckpt",

    "path_colab_training": "/content/Imagenet",
    "model_path": to_namedtuple('ModelPath', {
        "colab": "/content/GDrive/My Drive/Colab Notebooks/trained_model",
        "local": "trained_model"
    }),
    "report_path": to_namedtuple('ModelPath', {
        "colab": "/content/GDrive/My Drive/Colab Notebooks/report",
        "local": "trained_model"
    }),
    "path_colab_DUPR_model": "/content/GDrive/My Drive/Colab Notebooks/trained_model",
    "path_colab_DUPR_report": "/content/GDrive/My Drive/Colab Notebooks/report",
    "path_local_training": ".",

    # Controllers
    "is_colab": False,
    "is_demo": True,

})

pascal_voc_conf = to_namedtuple('PascalVocConfig', {
    # todo: add dataset and update the <PascalVoc/>

    "training_dataset": to_namedtuple('TrainingDataset', {
        "batch_size": 8,
        "num_workers": 2
    }),

    "test_dataset": to_namedtuple('TestDataset', {
        "batch_size": 4,
        "num_workers": 2
    }),

    "optimizer": to_namedtuple('Optimizer', {
        "lr": 0.005,
        "gamma": 0.1,
        "step": 1,
        "weight_decay": 0.0005
    }),

})
