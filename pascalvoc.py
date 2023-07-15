import os
import sys

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import pascal_voc_conf, pretraining_conf
from dataloader import PascalVoc
from nets.pascalvoc_trainer import evaluate, get_object_detection_model, train_one_epoch
from utils import collate_fn
from utils.plotters import plot_img_bbox
from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD
import torch
from torchmetrics.detection import MeanAveragePrecision

if __name__ == '__main__':
    dataset = PascalVoc(mode="train").dataset
    # plot the sample bbox
    img, target = dataset[4321]
    plot_img_bbox(img, target)

    # NOTE: select subset of dataset for demo
    dataset_train_val = Subset(dataset, list(range(1000)))
    dataset_test = Subset(dataset, list(range(1000, 1500)))

    dataset_train_val_loader = DataLoader(
            dataset_train_val,
            batch_size=pascal_voc_conf.training_dataset.batch_size,
            num_workers=pascal_voc_conf.training_dataset.num_workers,
            collate_fn=collate_fn,
            shuffle=True)

    dataset_test_loader = DataLoader(
            dataset_test,
            batch_size=pascal_voc_conf.test_dataset.batch_size,
            num_workers=pascal_voc_conf.test_dataset.num_workers,
            collate_fn=collate_fn,
            shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dupr_local_path = os.path.join(sys.path[0], pretraining_conf.model_path.local)
    dupr_model_path = pretraining_conf.model_path.colab if pretraining_conf.is_colab else dupr_local_path

    # NOTE: download the latest version checkpoint in colab
    # TODO: must be hoisted to the top of colab cell
    # !gdown https://drive.google.com/u/0/uc?id=1-c8ZJbhMX0w5FQR5-lshwK9yZAbhUGJm&export=download

    object_detector_model = get_object_detection_model(
            dupr_model_path,
            pretraining_conf.checkpoint,
            num_classes=len(PascalVoc.CLASSES_NAMES)
    )
    object_detector_model.to(device)

    model_params_count = sum(p.numel() for p in object_detector_model.parameters() if p.requires_grad)
    print(f"Faster R-CNN resnet-50 has {model_params_count} parameters.")

    params = [p for p in object_detector_model.parameters() if p.requires_grad]

    optimizer = SGD(
            params,
            lr=pascal_voc_conf.optimizer.lr,
            momentum=pascal_voc_conf.optimizer.momentum,
            weight_decay=pascal_voc_conf.optimizer.weight_decay,
    )
    lr_scheduler = StepLR(
            optimizer,
            step_size=pascal_voc_conf.optimizer.step,
            gamma=pascal_voc_conf.optimizer.gamma,
    )

    metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.75])
    for epoch in tqdm(range(pretraining_conf.epochs), 'epochs', leave=False):
        loss = train_one_epoch(object_detector_model, optimizer, dataset_train_val_loader, device)
        lr_scheduler.step()
        grand_truth, preds = evaluate(object_detector_model, dataset_test_loader, device=device)
        metric.update(preds, grand_truth)
        mAP = metric.compute()

        print(
                "mAP = ", mAP['map'].detach().item(),
                " map_50 = ", mAP['map_50'].detach().item(),
                " map_75 = ", mAP['map_75'].detach().item()
        )
