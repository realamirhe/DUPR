import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import RoIAlign


class DUPRModel(nn.Module):
    def __init__(self, sampleBatch=32, dim=128, K=65536, m=0.999, fm=4):
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
        self.sampleBatch = sampleBatch

        self.backbone_f1 = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone_f2 = resnet50(weights=ResNet50_Weights.DEFAULT)

        # freeze backbones
        """for param_f1, param_f2 in zip(
                self.backbone_f1.parameters(),
                self.backbone_f2.parameters()
        ):
            param_f1.requires_grad = False
            param_f2.requires_grad = False"""

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
        # replace the keys at ptr (dequeue and enqueue)
        temp_keys = torch.flatten(keys.permute(1, 0, 2), 1)
        indices = np.random.choice(a=batch_size * patch_size, size=self.sampleBatch)
        samples = temp_keys[:, indices]
        if (ptr + self.sampleBatch >= self.K):
            ptr = 0
        self.queue_patch[:, ptr:ptr + self.sampleBatch, fm] = samples
        ptr = ptr + self.sampleBatch  # move pointer
        self.queue_patch_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_image(self, defult_batch_size, keys, fm):
        # gather keys before updating queue
        # keys = self.concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_image_ptr)
        # assert self.K % batch_size == 0

        # replace the keys at ptr (dequeue and enqueue)
        if (ptr + defult_batch_size >= self.K):
            ptr = 0  # move pointer
        self.queue_image[:, ptr:ptr + batch_size, fm] = keys.T

        ptr = ptr + batch_size
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
