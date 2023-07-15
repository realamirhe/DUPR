import os

import pandas as pd
import torch
from torch import nn, optim
from tqdm import tqdm

from deeplearning import AverageMeter
from nets.dupr_model import DUPRModel


class DUPRTrainer:
    def __init__(self, batch_size,
                 alpha=[0.1, 0.4, 0.7, 1.0],
                 beta=[0, 0, 1, 1],
                 dim=128,
                 K=65536,
                 m=0.999,
                 T=0.07
                 ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.alpha = alpha
        self.beta = beta
        self.dim = dim
        self.K = K
        self.m = m
        self.T = T
        self.batch_size = batch_size
        self.model = DUPRModel(dim=dim, K=K, m=m)

    def make_optimizer(self, learning_rate, gamma, step_size):
        self.optimizer = optim.Adam(
                [param for param in self.model.parameters() if param.requires_grad == True],
                lr=learning_rate
        )
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.step_size = step_size

    def patch_loss(self, q, k, fm):
        self.model._momentum_update_key_encoder()
        # l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        k = k.view(*(k.size()[0:2]), -1)
        q = q.view(*(q.size()[0:2]), -1)
        l_pos = torch.einsum('bnc,bnc->bc', [q, k]).unsqueeze(-1)
        # l_pos = torch.bmm(q, k)

        # negative logits: NxK
        keys = self.model.queue_patch.clone().detach()[:, :, fm]
        l_neg = torch.einsum('bnc,dk->bck', [q, keys])
        # l_neg = torch.dot(q, self.model.queue_patch.clone().detach())
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=2)
        # apply temperature
        logits /= self.T
        # labels: positive key indicators
        labels = torch.zeros((logits.shape[0], logits.shape[-1]), dtype=torch.long, device=self.device)
        # dequeue and enqueue
        if (self.model.training):
            self.model._dequeue_and_enqueue_patch(k, fm)

        return logits, labels

    def image_loss(self, q, k, fm):
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        keys = self.model.queue_patch.clone().detach()[:, :, fm]
        l_neg = torch.einsum('nc,ck->nk', [q, keys])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.T
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
        # dequeue and enqueue
        if (self.model.training):
            self.model._dequeue_and_enqueue_image(self.batch_size, k, fm)

        return logits, labels

    def train(self, train_loader, val_loader, epochs,
              ckpt_save_freq, ckpt_save_path, report_path):
        model_file_name = ""
        report_file_name = ""
        self.model = self.model.to(self.device)

        # loss function
        criterion = nn.CrossEntropyLoss()

        lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

        report = pd.DataFrame(columns=["mode", "epoch", "learning_rate", "batch_size",
                                       "batch_index", "loss_batch", "avg_train_loss_till_current_batch",
                                       "avg_train_top1_acc_till_current_batch", "avg_val_loss_till_current_batch",
                                       "avg_val_top1_acc_till_current_batch"]
                              )

        for epoch in tqdm(range(1, epochs + 1)):
            # top1_acc_train = AverageMeter()
            loss_avg_train = AverageMeter()
            # top1_acc_val = AverageMeter()
            loss_avg_val = AverageMeter()

            self.model.train()
            mode = "train"

            loop_train = tqdm(enumerate(train_loader, 1), total=len(train_loader),
                              desc="train", position=0, leave=True)
            for batch_idx, (I1, B1, I2, B2) in loop_train:
                I1 = I1.to(self.device).float()
                I2 = I2.to(self.device).float()
                B1 = B1.to(self.device).float()
                B2 = B2.to(self.device).float()

                for i in range(B1.size()[0]):
                    B1[i, 0] = i
                    B2[i, 0] = i

                r1, r2, v1, v2 = self.model(I1, I2, B1, B2)

                loss = 0
                loss_image_sigma = 0
                loss_patch_sigma = 0
                for i in range(4):
                    patch_logits, patch_labels = self.patch_loss(r1[i], r2[i], i)
                    image_logits, image_labels = self.image_loss(v1[i], v2[i], i)

                    loss_patch_m = criterion(patch_logits, patch_labels)
                    loss_image_m = criterion(image_logits, image_labels)
                    loss_image_sigma += self.alpha[i] * loss_image_m
                    loss_patch_sigma += self.beta[i] * loss_patch_m

                loss = loss_image_sigma + loss_patch_sigma

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # acc1 = self.accuracy(labels_pred, labels)
                # top1_acc_train.update(acc1[0], images.size(0))
                loss_avg_train.update(loss.detach().item(), I1.size(0))

                new_row = pd.DataFrame(
                        {"mode": mode, "epoch": epoch,
                         "learning_rate": self.optimizer.param_groups[0]["lr"],
                         "batch_size": I1.size(0), "batch_index": batch_idx,
                         "loss_batch": loss.detach().item(),
                         "avg_train_loss_till_current_batch": loss_avg_train.avg,
                         "avg_train_top1_acc_till_current_batch": None,
                         "avg_val_loss_till_current_batch": None,
                         "avg_val_top1_acc_till_current_batch": None}, index=[0]
                )

                report.loc[len(report)] = new_row.values[0]

                loop_train.set_description(f"Train-iter: {epoch}")
                loop_train.set_postfix(
                        loss_batch="{:.4f}".format(loss.detach().item()),
                        train_loss="{:.4f}".format(loss_avg_train.avg),
                        # train_acc="{:.4f}".format(top1_acc_train.avg),
                        max_len=2, refresh=True
                )

            if epoch % ckpt_save_freq == 0:
                model_file_name = f"ckpt_DUPR_epoch{epoch}.ckpt"
                self.save_model(file_path=ckpt_save_path,
                                file_name=model_file_name,
                                model=self.model, optimizer=self.optimizer,
                                )

            self.model.eval()
            mode = "val"
            with torch.no_grad():
                loop_val = tqdm(enumerate(val_loader, 1),
                                total=len(val_loader), desc="val", position=0, leave=True,
                                )
                for batch_idx, (I1, B1, I2, B2) in loop_val:
                    self.optimizer.zero_grad()
                    I1 = I1.to(self.device).float()
                    I2 = I2.to(self.device).float()
                    B1 = B1.to(self.device).float()
                    B2 = B2.to(self.device).float()
                    r1, r2, v1, v2 = self.model(I1, I2, B1, B2)

                    loss = 0
                    loss_image_sigma = 0
                    loss_patch_sigma = 0
                    for i in range(4):
                        patch_logits, patch_labels = self.patch_loss(r1[i], r2[i], i)
                        image_logits, image_labels = self.image_loss(v1[i], v2[i], i)

                        loss_patch_m = criterion(patch_logits, patch_labels)
                        loss_image_m = criterion(image_logits, image_labels)
                        loss_image_sigma += self.alpha[i] * loss_image_m
                        loss_patch_sigma += self.beta[i] * loss_patch_m

                    loss = loss_image_sigma + loss_patch_sigma
                    # acc1 = self.accuracy(labels_pred, labels)
                    # top1_acc_val.update(acc1[0], images.size(0))
                    loss_avg_val.update(loss.detach().item(), I1.size(0))

                    new_row = pd.DataFrame(
                            {"mode": mode,
                             "epoch": epoch,
                             "learning_rate": self.optimizer.param_groups[0]["lr"],
                             "batch_size": I1.size(0),
                             "batch_index": batch_idx,
                             "loss_batch": loss.detach().item(),
                             "avg_train_loss_till_current_batch": None,
                             "avg_train_top1_acc_till_current_batch": None,
                             "avg_val_loss_till_current_batch": loss_avg_val.avg,
                             "avg_val_top1_acc_till_current_batch": None}, index=[0]
                    )
                    report.loc[len(report)] = new_row.values[0]

                    loop_val.set_description(f"val-iter: {epoch}")
                    loop_val.set_postfix(
                            loss_batch="{:.4f}".format(loss.detach().item()),
                            val_loss="{:.4f}".format(loss_avg_val.avg),
                            # val_accuracy="{:.4f}".format(top1_acc_val.avg),
                            refresh=True,
                    )
            lr_scheduler.step()

        report_file_name = f"report.csv"
        os.path.join(report_path, report_file_name)
        report.to_csv(os.path.join(report_path, report_file_name))

        return model_file_name, report_file_name

    def load_model(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint["model"])

        if ("optimizer" in checkpoint.keys()):
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.model = self.model.to(self.device)

    @staticmethod
    def save_model(file_path, file_name, model, optimizer=None):
        state_dict = dict()
        state_dict["model"] = model.state_dict()

        if optimizer is not None:
            state_dict["optimizer"] = optimizer.state_dict()
        torch.save(state_dict, os.path.join(file_path, file_name))
