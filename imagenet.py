import os

import pandas as pd
import torch
from matplotlib import pyplot as plt

from config import pretraining_conf
from dataloader import ImagesDataset
from nets.duprtrainer import DUPRTrainer

if __name__ == '__main__':
    dataset_path = os.path.join(
            pretraining_conf.dataset.output,
            pretraining_conf.dataset.name
    )
    batch_size = pretraining_conf.training.batch_size

    train_dataset = ImagesDataset(dataset_path, mode='train')
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
    )

    val_dataset = ImagesDataset(dataset_path, mode='val')
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
    )

    # K is multiply of 263
    K_scale = pretraining_conf.model.negative_keys_scale  # 8
    model_trainer = DUPRTrainer(
            batch_size=batch_size,
            dim=pretraining_conf.model.feature_dim,  # 128
            K=((196 + 49 + 9 + 9) * 32) * K_scale,  # S_sums
            m=pretraining_conf.model.moco_momemntum,  # 0.999,
            T=pretraining_conf.model.softmax_temperature,  # 0.07
    )

    model_trainer.make_optimizer(
            pretraining_conf.optimizer.lr,
            pretraining_conf.optimizer.gamma,
            pretraining_conf.optimizer.step
    )

    model_path = pretraining_conf.model_path
    report_path = pretraining_conf.report_path
    model_fileName, report_fileName = model_trainer.train(
            train_loader,
            val_loader,
            epochs=pretraining_conf.model.epochs,
            ckpt_save_freq=pretraining_conf.model.save_freq,
            ckpt_save_path=pretraining_conf.model_path,
            report_path=pretraining_conf.report_path
    )

    report = pd.read_csv(os.path.join(report_path, report_fileName))
    train_report = report[report['mode'] == "train"].groupby("epoch").last()
    val_report = report[report['mode'] == "val"].groupby("epoch").last()

    plt.title('Loss')
    plt.plot(train_report["avg_train_loss_till_current_batch"], color='blue', label='train')
    plt.plot(val_report["avg_val_loss_till_current_batch"], color='orange', label='validation')
    plt.legend(('train', 'validation'), loc='upper left')
    plt.show()
