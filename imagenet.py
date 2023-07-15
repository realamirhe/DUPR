import os
import sys

import pandas as pd
import torch
from matplotlib import pyplot as plt

from config import training_config
from dataloader import ImagesDS
from nets.duprtrainer import DUPRTrainer
from utils import load_files

if __name__ == '__main__':
    dataset_path = "imagenette2-320"
    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "val")
    train_path = os.path.join(sys.path[0], train_path)
    val_path = os.path.join(sys.path[0], val_path)

    train_images, val_images = load_files(train_path, val_path)
    print(len(train_images))
    train_DS = ImagesDS(train_images)
    val_DS = ImagesDS(val_images)

    batch_size = training_config.batch_size
    train_loader = torch.utils.data.DataLoader(train_DS, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_DS, batch_size=batch_size, shuffle=True)

    S_sums = ((196 + 49 + 9 + 9) * 32) * batch_size
    model_trainer = DUPRTrainer(batch_size=batch_size, dim=128, K=S_sums, m=0.999, T=0.07)  # K is multiply of 263
    model_trainer.make_optimizer(0.0001, 0.5, 1)

    if(training_config.is_colab):
        # TODO: !mkdir -p GDrive/My Drive/Colab Notebooks/training_model
        # TODO: !mkdir -p GDrive/My Drive/Colab Notebooks/report
        model_path = os.path.join('GDrive/My Drive/Colab Notebooks', 'trained_model')
        report_path = os.path.join('GDrive/My Drive/Colab Notebooks', 'report')
    else:
        # TODO: !mkdir -p training_model
        # TODO: !mkdir -p report
         model_path = os.path.join(sys.path[0],'trained_model')
         report_path = os.path.join(sys.path[0], 'report')
    

    model_fileName, report_fileName = model_trainer.train(
            train_loader,
            val_loader,
            epochs=training_config.epochs,
            ckpt_save_freq=training_config.epochs,
            ckpt_save_path=model_path,
            report_path=report_path
    )

    report = pd.read_csv(os.path.join(report_path, report_fileName))
    train_report = report[report['mode'] == "train"].groupby("epoch").last()
    val_report = report[report['mode'] == "val"].groupby("epoch").last()

    # plot loss
    plt.title('Loss')
    plt.plot(train_report["avg_train_loss_till_current_batch"], color='blue', label='train')
    plt.plot(val_report["avg_val_loss_till_current_batch"], color='orange', label='validation')
    plt.legend(('train', 'validation'), loc='upper left')
    plt.show()

    model_path = os.path.join(training_config.base_path, 'trained_model')
