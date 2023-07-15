import glob
import os
import sys
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import RoIAlign
from tqdm import tqdm
from config import training_config


def load_files(train_path, val_path, source=None):
    if (source == "GoogleDrive"):
        drive.mount('/content/GDrive/')
        tgz_file_path="GDrive/My Drive/Colab Notebooks/imagenette2-320.tgz"
        file = tarfile.open(tgz_file_path)
        # extracting a specific file
        file.extractall()
        file.close()
        
        path="/content/imagenette2-320"
        train_path = os.path.join(path, train_path)
        val_path = os.path.join(path, val_path)
    if source is None:
        
        path = os.path.join(sys.path[0],"imagenette2-320")
        train_path = os.path.join(path, train_path)
        val_path = os.path.join(path, val_path)
    classes = os.listdir(train_path)
    print("Total Classes: ", len(classes))

    train_imgs = []
    val_imgs = []
    for _class in classes:
        train_imgs += glob.glob(os.path.join(train_path, _class) + '/*.JPEG', recursive=True)
        val_imgs += glob.glob(os.path.join(val_path, _class) + '/*.JPEG', recursive=True)

    print("Total train images: ", len(train_imgs))
    print("Total test images: ", len(val_imgs))

    return train_imgs, val_imgs


class ImagesDS(Dataset):
    def __init__(self, imgs_list):
        super().__init__()
        self.imgs_list = imgs_list

    def crop_intersection_transform(self, image, index):
        x, y = image.size
        p = np.random.rand()
        if p <= 0.5:
            l1 = 0
            t1 = 0
            h1 = np.random.randint(int(y / 2), int(3 * y / 4))
            w1 = np.random.randint(int(x / 2), int(3 * x / 4))
            l2 = np.random.randint(int(x / 4), int(x / 2))
            t2 = np.random.randint(int(y / 4), int(y / 2))
            h2 = y - t2
            w2 = x - l2
        else:
            l1 = 0
            h1 = np.random.randint(int(y / 2), int(3 * y / 4))
            t1 = y - h1
            w1 = np.random.randint(int(x / 2), int(3 * x / 4))
            l2 = np.random.randint(int(x / 4), int(x / 2))
            t2 = 0
            w2 = x - l2
            h2 = np.random.randint(int(y / 2), int(3 * y / 4))

        tb = max([t1, t2])
        if tb == t1:
            tb1 = 0
            tb2 = t1 - t2
        else:
            tb2 = 0
            tb1 = t2 - t1
        lb = max([l1, l2])
        if lb == l1:
            lb1 = 0
            lb2 = l1 - l2
        else:
            lb2 = 0
            lb1 = l2 - l1
        hb = min([t1 + h1, t2 + h2]) - tb
        wb = min([l1 + w1, l2 + w2]) - lb

        # crop top, left, height, width
        t1 = TF.crop(image, t1, l1, h1, w1)
        t2 = TF.crop(image, t2, l2, h2, w2)
        # B = TF.crop(image, tb, lb, hb, wb)

        trf1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([224, 224], antialias=True),
            transforms.ColorJitter()
        ])

        trf2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([224, 224], antialias=True),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(p=1)
        ])

        I1 = trf1(t1)
        I1_x_ratio = I1.size(1) / t1.size[0]
        I1_y_ratio = I1.size(2) / t1.size[1]
        tb1 = tb1 * I1_y_ratio
        lb1 = lb1 * I1_x_ratio

        rb1 = lb1 + wb * I1_x_ratio
        bb1 = tb1 + hb * I1_y_ratio

        I2 = trf2(t2)
        I2_x_ratio = I2.size(1) / t2.size[0]
        I2_y_ratio = I2.size(2) / t2.size[1]

        rotated_left = t2.size[1] - lb2 - wb
        rotated_right = t2.size[1] - lb2
        rotated_top = tb2
        rotated_bottom = tb2 + hb

        rotated_left *= I2_x_ratio
        rotated_right *= I2_x_ratio
        rotated_top *= I2_y_ratio
        rotated_bottom *= I2_y_ratio

        lb2, rb2, tb2, bb2 = rotated_left, rotated_right, rotated_top, rotated_bottom

        B1 = torch.Tensor([0, lb1, tb1, rb1, bb1])
        B2 = torch.Tensor([0, lb2, tb2, rb2, bb2])

        return I1, B1, I2, B2

    def __getitem__(self, index):
        image_path = self.imgs_list[index]
        image = Image.open(image_path)
        if (image.mode == "L"):
            image = image.convert("RGB")
        I1, B1, I2, B2 = self.crop_intersection_transform(image, index)

        return I1, B1, I2, B2

    def __len__(self):
        return len(self.imgs_list)


class AverageMeter(object):
    """
    computes and stores the average and current value
    """

    def __init__(self, start_val=0, start_count=0, start_avg=0, start_sum=0):
        self.reset()
        self.val = start_val
        self.avg = start_avg
        self.sum = start_sum
        self.count = start_count

    def reset(self):
        """
        Initialize 'value', 'sum', 'count', and 'avg' with 0.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        """
        Update 'value', 'sum', 'count', and 'avg'.
        """
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


class DUPR_Model(nn.Module):
    def __init__(self, dim=128, K=65536, m=0.999, fm=4):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()

        self.dim = dim
        self.K = K
        self.m = m
        self.fm = fm

        self.backbone_f1 = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone_f2 = resnet50(weights=ResNet50_Weights.DEFAULT)

        # freeze backbones
        for param_f1, param_f2 in zip(
                self.backbone_f1.parameters(),
                self.backbone_f2.parameters()
        ):
            param_f1.requires_grad = False
            param_f2.requires_grad = False

        return_nodes = {
            'layer1': 'layer1',
            'layer2': 'layer2',
            'layer3': 'layer3',
            'layer4': 'layer4',
        }
        self.backbone_f1 = create_feature_extractor(self.backbone_f1, return_nodes=return_nodes)
        self.backbone_f2 = create_feature_extractor(self.backbone_f2, return_nodes=return_nodes)

        # create the encoder 1
        self.roi1 = RoIAlign((3, 3), 1 / 4, -1, True)
        self.roi2 = RoIAlign((3, 3), 1 / 8, -1, True)
        self.roi3 = RoIAlign((14, 14), 1 / 16, -1, True)
        self.roi4 = RoIAlign((7, 7), 1 / 32, -1, True)

        self.g1 = nn.Conv2d(256, dim, (1 * 1))
        self.g2 = nn.Conv2d(512, dim, (1 * 1))
        self.g3 = nn.Conv2d(1024, dim, (1 * 1))
        self.g4 = nn.Conv2d(2048, dim, (1 * 1))

        # create the encoder 2
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.h1 = nn.Linear(256, dim)
        self.h2 = nn.Linear(512, dim)
        self.h3 = nn.Linear(1024, dim)
        self.h4 = nn.Linear(2048, dim)

        # create the queue
        self.register_buffer("queue_patch", torch.randn(dim, K, fm))
        self.queue_patch = nn.functional.normalize(self.queue_patch, dim=0)

        self.register_buffer("queue_patch_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_image", torch.randn(dim, K, fm))
        self.queue_image = nn.functional.normalize(self.queue_image, dim=0)

        self.register_buffer("queue_image_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue_patch(self, keys, fm):
        # gather keys before updating queue
        # keys = self.concat_all_gather(keys)

        batch_size = keys.shape[0]
        patch_size = keys.shape[2]

        ptr = int(self.queue_patch_ptr)
        #assert self.K % (batch_size * patch_size) == 0

        # replace the keys at ptr (dequeue and enqueue)
        tt = torch.flatten(keys.permute(1, 0, 2), 1)
        self.queue_patch[:, ptr:ptr + batch_size * patch_size, fm] = torch.flatten(keys.permute(1, 0, 2), 1)
        ptr = (ptr + batch_size * patch_size) % self.K  # move pointer

        self.queue_patch_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_image(self, keys, fm):
        # gather keys before updating queue
        # keys = self.concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_image_ptr)
        #assert self.K % batch_size == 0

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_image[:, ptr:ptr + batch_size, fm] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_image_ptr[0] = ptr

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.backbone_f1.parameters(), self.backbone_f2.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, img1, img2, b1, b2):

        # encode image using encoder 1 and encoder 2
        f1_output = self.backbone_f1(img1)
        f2_output = self.backbone_f2(img2)

        # forward patch level
        roi1_m1 = self.roi1(f1_output["layer1"], b1)
        roi1_m2 = self.roi2(f1_output["layer2"], b1)
        roi1_m3 = self.roi3(f1_output["layer3"], b1)
        roi1_m4 = self.roi4(f1_output["layer4"], b1)

        r1 = [
            nn.functional.normalize(self.g1(roi1_m1), dim=1),
            nn.functional.normalize(self.g2(roi1_m2), dim=1),
            nn.functional.normalize(self.g3(roi1_m3), dim=1),
            nn.functional.normalize(self.g4(roi1_m4), dim=1)
        ]

        roi2_m1 = self.roi1(f2_output["layer1"], b2)
        roi2_m2 = self.roi2(f2_output["layer2"], b2)
        roi2_m3 = self.roi3(f2_output["layer3"], b2)
        roi2_m4 = self.roi4(f2_output["layer4"], b2)

        r2 = [
            nn.functional.normalize(self.g1(roi2_m1), dim=1),
            nn.functional.normalize(self.g2(roi2_m2), dim=1),
            nn.functional.normalize(self.g3(roi2_m3), dim=1),
            nn.functional.normalize(self.g4(roi2_m4), dim=1),
        ]

        # forward image level
        gap1_m1 = self.gap(f1_output["layer1"])
        gap1_m2 = self.gap(f1_output["layer2"])
        gap1_m3 = self.gap(f1_output["layer3"])
        gap1_m4 = self.gap(f1_output["layer4"])

        gap2_m1 = self.gap(f2_output["layer1"])
        gap2_m2 = self.gap(f2_output["layer2"])
        gap2_m3 = self.gap(f2_output["layer3"])
        gap2_m4 = self.gap(f2_output["layer4"])

        v1 = [
            nn.functional.normalize(self.h1(torch.squeeze(gap1_m1, (2, 3))), dim=1),
            nn.functional.normalize(self.h2(torch.squeeze(gap1_m2, (2, 3))), dim=1),
            nn.functional.normalize(self.h3(torch.squeeze(gap1_m3, (2, 3))), dim=1),
            nn.functional.normalize(self.h4(torch.squeeze(gap1_m4, (2, 3))), dim=1)]

        v2 = [
            nn.functional.normalize(self.h1(torch.squeeze(gap2_m1, (2, 3))), dim=1),
            nn.functional.normalize(self.h2(torch.squeeze(gap2_m2, (2, 3))), dim=1),
            nn.functional.normalize(self.h3(torch.squeeze(gap2_m3, (2, 3))), dim=1),
            nn.functional.normalize(self.h4(torch.squeeze(gap2_m4, (2, 3))), dim=1)]

        return r1, r2, v1, v2


class DUPR_trainer:
    def __init__(self,
                 alpha=[0.1, 0.4, 0.7, 1.0],
                 beta=[0, 0, 1, 1],
                 dim=128,
                 K=65536,
                 m=0.999,
                 T=0.07
                 ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        self.alpha = alpha
        self.beta = beta
        self.dim = dim
        self.K = K
        self.m = m
        self.T = T
        self.model = DUPR_Model(dim=dim, K=K, m=m)

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
        k = k.view(*(k.size()[0:2]), -1)
        q = q.view(*(q.size()[0:2]), -1)
        l_pos = torch.einsum('bnc,bnc->bc', [q, k]).unsqueeze(-1)

        # negative logits
        keys = self.model.queue_patch.clone().detach()[:, :, fm]
        l_neg = torch.einsum('bnc,dk->bck', [q, keys])
        # logits
        logits = torch.cat([l_pos, l_neg], dim=2)
        # apply temperature
        logits /= self.T
        # labels: positive key indicators
        labels = torch.zeros((logits.shape[0],logits.shape[-1]), dtype=torch.long, device=self.device)
        # dequeue and enqueue
        if(self.model.training):
            self.model._dequeue_and_enqueue_patch(k, fm)

        return logits, labels

    def image_loss(self, q, k, fm):
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits
        keys = self.model.queue_patch.clone().detach()[:, :, fm]
        l_neg = torch.einsum('nc,ck->nk', [q, keys])
        # logits
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.T
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
        # dequeue and enqueue
        if(self.model.training):
            self.model._dequeue_and_enqueue_image(k, fm)

        return logits, labels

    def train(self, train_loader, val_loader, epochs,
              ckpt_save_freq, ckpt_save_path, report_path):
        model_fileName = ""
        report_fileName = ""
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
            loss_avg_train = AverageMeter()
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
                    B1[i,0]=i
                    B2[i,0]=i

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
                loss_avg_train.update(loss.detach().item(), I1.size(0))

                new_row = pd.DataFrame(
                        {"mode": mode, "epoch": epoch,
                         "learning_rate": self.optimizer.param_groups[0]["lr"],
                         "batch_size": I1.size(0), "batch_index": batch_idx,
                         "loss_batch": loss.detach().item(),
                         "avg_train_loss_till_current_batch": loss_avg_train.avg,
                         "avg_val_loss_till_current_batch": None}, index=[0]
                )

                report.loc[len(report)] = new_row.values[0]

                loop_train.set_description(f"Train-iter: {epoch}")
                loop_train.set_postfix(
                        loss_batch="{:.4f}".format(loss.detach().item()),
                        train_loss="{:.4f}".format(loss_avg_train.avg),
                        max_len=2, refresh=True
                )

            if epoch % ckpt_save_freq == 0:
                model_fileName = f"ckpt_DUPR_epoch{epoch}.ckpt"
                self.save_model(file_path=ckpt_save_path,
                                file_name=model_fileName,
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
                    loss_avg_val.update(loss.detach().item(), I1.size(0))

                    new_row = pd.DataFrame(
                            {"mode": mode,
                             "epoch": epoch,
                             "learning_rate": self.optimizer.param_groups[0]["lr"],
                             "batch_size": I1.size(0),
                             "batch_index": batch_idx,
                             "loss_batch": loss.detach().item(),
                             "avg_train_loss_till_current_batch": None,
                             "avg_val_loss_till_current_batch": loss_avg_val.avg,
                             }, index=[0]
                    )
                    report.loc[len(report)] = new_row.values[0]

                    loop_val.set_description(f"val-iter: {epoch}")
                    loop_val.set_postfix(
                            loss_batch="{:.4f}".format(loss.detach().item()),
                            val_loss="{:.4f}".format(loss_avg_val.avg),
                            refresh=True,
                    )
            lr_scheduler.step()

        report_fileName = f"report.csv"
        os.path.join(report_path, report_fileName)
        report.to_csv(os.path.join(report_path, report_fileName))

        return model_fileName, report_fileName

    def load_model(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint["model"])

        if ("optimizer" in checkpoint.keys()):
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.model = self.model.to(self.device)

    def save_model(self, file_path, file_name, model, optimizer=None):

        state_dict = dict()
        state_dict["model"] = model.state_dict()

        if optimizer is not None:
            state_dict["optimizer"] = optimizer.state_dict()
        torch.save(state_dict, os.path.join(file_path, file_name))


def to_namedtuple(name, dictionary):
    return namedtuple(name, dictionary)(**dictionary)


config = to_namedtuple('TraningArg', {
    "base_path": ".",

    # model configs
    "epochs": 2,
    "batch_size": 4
})

if __name__ == '__main__':
    dataset_path = "imagenette2-320"
    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "val")
    train_path = os.path.join(sys.path[0], train_path)
    val_path = os.path.join(sys.path[0], val_path)

    train_imgs, val_imgs = load_files(train_path, val_path)

    train_DS = ImagesDS(train_imgs[0:100])
    val_DS = ImagesDS(val_imgs[0:10])

    batch_size = config.batch_size
    train_loader = torch.utils.data.DataLoader(train_DS, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_DS, batch_size=batch_size, shuffle=True)

    S_sums = ((196 + 49 + 9 + 9) * 15)*batch_size
    model_trainer = DUPR_trainer(dim=128, K=S_sums, m=0.999, T=0.07)  # K is multiply of 263
    model_trainer.make_optimizer(0.0001, 0.5, 1)

    model_path = os.path.join(config.base_path, 'trained_model')
    report_path = os.path.join(config.base_path, 'report')

    model_fileName, report_fileName = model_trainer.train(
            train_loader,
            val_loader,
            epochs=config.epochs,
            ckpt_save_freq=5,
            ckpt_save_path=model_path,
            report_path=report_path
    )

    report_path = os.path.join(sys.path[0], 'report')
    report = pd.read_csv(os.path.join(report_path, report_fileName))
    train_report = report[report['mode'] == "train"].groupby("epoch").last()
    val_report = report[report['mode'] == "val"].groupby("epoch").last()

    # plot loss
    plt.title('Loss')
    plt.plot(train_report["avg_train_loss_till_current_batch"], color='blue', label='train')
    plt.plot(val_report["avg_val_loss_till_current_batch"], color='orange', label='validation')
    plt.legend(('train', 'validation'), loc='upper left')
    plt.show()

    model_path = os.path.join(config.base_path, 'trained_model')
