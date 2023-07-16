from utils import to_namedtuple

BASE_PATH = '/content/DUPR' if 'google.colab' in str(get_ipython()) else '.'

pretraining_conf = to_namedtuple('PreTrainingConfig', {
    # Configuration
    "version": "0.0.1",

    "training": to_namedtuple('Training', {
        "batch_size": 8,
    }),

    "optimizer": to_namedtuple('Optimizer', {
        "lr": 0.0001,
        "gamma": 0.5,
        "step": 1
    }),

    "model": to_namedtuple('Model', {
        "negative_keys_scale": 8,
        "feature_dim": 128,
        "moco_momemntum": 0.999,
        "softmax_temperature": 0.07,
        "epochs": 25,
        "save_freq": 25,
    }),

    "dataset": to_namedtuple('Dataset', {
        "name": "imagenette2-320",
        "url": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
        "output": f"{BASE_PATH}/datasets"
    }),

    "model_path": f"{BASE_PATH}/trained_model/imagenet",
    "report_path": f"{BASE_PATH}/report/imagenet",

    # Controllers
    "is_colab": 'google.colab' in str(get_ipython()),
    "is_demo": True,
})

pascal_voc_conf = to_namedtuple('PascalVocConfig', {
    "version": "0.0.1",

    "model_path": f"{BASE_PATH}/trained_model/pascal/",
    "report_path": f"{BASE_PATH}/report/pascal/",
    "model_output": f"{BASE_PATH}/outputs/object-detections",

    'model': to_namedtuple('Model', {
        "download_url": "https://drive.google.com/u/0/uc?id=1-c8ZJbhMX0w5FQR5-lshwK9yZAbhUGJm&export=download",
        "checkpoint": "ckpt_DUPR_epoch20.ckpt",
        "epochs": 20,
    }),

    "dataset": to_namedtuple('Dataset', {
        "name": "pascal-voc",
        "output": f"{BASE_PATH}/datasets/pascal-voc"
    }),

    "training_dataset": to_namedtuple('TrainingDataset', {
        "batch_size": 8,
        "num_workers": 2,
    }),

    "test_dataset": to_namedtuple('TestDataset', {
        "batch_size": 4,
        "num_workers": 2
    }),

    "optimizer": to_namedtuple('Optimizer', {
        "lr": 0.005,
        "gamma": 0.1,
        "step": 1,
        "momentum": 0.999,
        "weight_decay": 0.0005
    }),
})
