from utils import to_namedtuple

training_config = to_namedtuple('TrainingArg', {
    "base_path": ".",
    "
    # todo: add dataset and update the <PascalVoc/>

    # model configs
    "epochs": 25,
    "batch_size_DUPR": 8,
    "batch_size_pascalVOC_train": 8,
    "batch_size_pascalVOC_test": 4,
    "DUPR_lr":0.0001,
    "DUPR_gamma":0.5,
    "DUPR_step":1

    
    "dupr_model_file_name":"ckpt_DUPR_epoch20.ckpt",
    "path_local_DUPR_model":"trained_model",
    "num_workers":2

    "is_colab": False,
    "path_colab_training": "/content/Imagenet",
    "path_colab_DUPR_model":"/content/GDrive/My Drive/Colab Notebooks/trained_model"
    "path_colab_DUPR_report":"/content/GDrive/My Drive/Colab Notebooks/report"
    "path_local_training": "."
    

})

# imagenet_config // pre-training
# pascalvoc_config // object detection
#
