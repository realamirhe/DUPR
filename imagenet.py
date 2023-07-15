import os
import sys

import pandas as pd
import torch
from matplotlib import pyplot as plt

from config import pretraining_conf
from dataloader import ImagesDataset
from nets.duprtrainer import DUPRTrainer
from utils import load_files

if __name__ == '__main__':
    # TODO: add dataset downloader and configuration
    dataset_path = "imagenette2-320"
    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "val")
    train_path = os.path.join(sys.path[0], train_path)
    val_path = os.path.join(sys.path[0], val_path)

    train_images, val_images = load_files(train_path, val_path)
    print(f"Loaded {len(train_images)} images")
    train_DS = ImagesDataset(train_images)
    val_DS = ImagesDataset(val_images)

    batch_size = pretraining_conf.batch_size_DUPR
    train_loader = torch.utils.data.DataLoader(train_DS, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_DS, batch_size=batch_size, shuffle=True)

    # K is multiply of 263
    K_scale = 8
    model_trainer = DUPRTrainer(
            batch_size=batch_size,
            dim=128,
            K=((196 + 49 + 9 + 9) * 32) * K_scale,  # S_sums
            m=0.999,
            T=0.07
    )
    model_trainer.make_optimizer(
            pretraining_conf.optimizer.lr,
            pretraining_conf.optimizer.gamma,
            pretraining_conf.optimizer.step
    )

    if pretraining_conf.is_colab:
        # TODO: !mkdir -p GDrive/My Drive/Colab Notebooks/training_model
        # TODO: !mkdir -p GDrive/My Drive/Colab Notebooks/report
        model_path = pretraining_conf.model_path.colab
        report_path = pretraining_conf.path_colab_DUPR_report
    else:
        # TODO: !mkdir -p training_model
        # TODO: !mkdir -p report
        # TODO: Remove the os.path.join(sys.path[0]
        model_path = os.path.join(sys.path[0], 'trained_model')
        report_path = os.path.join(sys.path[0], 'report')

    model_fileName, report_fileName = model_trainer.train(
            train_loader,
            val_loader,
            epochs=pretraining_conf.epochs,
            ckpt_save_freq=pretraining_conf.save_freq,
            ckpt_save_path=model_path,
            report_path=report_path
    )

    report = pd.read_csv(os.path.join(report_path, report_fileName))
    train_report = report[report['mode'] == "train"].groupby("epoch").last()
    val_report = report[report['mode'] == "val"].groupby("epoch").last()

    plt.title('Loss')
    plt.plot(train_report["avg_train_loss_till_current_batch"], color='blue', label='train')
    plt.plot(val_report["avg_val_loss_till_current_batch"], color='orange', label='validation')
    plt.legend(('train', 'validation'), loc='upper left')
    plt.show()
