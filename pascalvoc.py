import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from config import pascal_voc_conf
from dataloader import PascalVoc
from nets.pascalvoc_trainer import evaluate, get_object_detection_model, train_one_epoch
from utils import collate_fn
from utils.plotters import plot_img_bbox

if __name__ == '__main__':
    dataset_train_val = PascalVoc(mode="train").data
    dataset_test = PascalVoc(mode="test").data

    # plot the sample bbox
    img, target = dataset_train_val[4321]
    plot_img_bbox(img, target)

    # For fast test and demo purpose
    # from torch.utils.data import  Subset
    # dataset_test = Subset(dataset_train_val, list(range(500, 1000)))
    # dataset_train_val = Subset(dataset_train_val, list(range(500)))

    # plot the sample bbox
    img, target = dataset_train_val[4321]
    plot_img_bbox(img, target)

    dataset_train_val_loader = DataLoader(
            dataset_train_val,
            batch_size=pascal_voc_conf.training_dataset.batch_size,
            num_workers=pascal_voc_conf.training_dataset.num_workers,
            collate_fn=collate_fn,
            shuffle=True
    )

    dataset_test_loader = DataLoader(
            dataset_test,
            batch_size=pascal_voc_conf.test_dataset.batch_size,
            num_workers=pascal_voc_conf.test_dataset.num_workers,
            collate_fn=collate_fn,
            shuffle=False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    object_detector_model = get_object_detection_model(
            pascal_voc_conf.model_path,
            pascal_voc_conf.model.checkpoint,
            num_classes=len(PascalVoc.CLASSES_NAMES),
            device=device,
            backbone='backbone'
    )
    object_detector_model.to(device)

    model_params_count = sum(p.numel() for p in object_detector_model.parameters() if p.requires_grad)
    print(f"Faster R-CNN resnet-50 has {'{:,}'.format(model_params_count)} parameters.")

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

    for epoch in tqdm(range(pascal_voc_conf.model.epochs), 'epochs', leave=False):
        loss = train_one_epoch(object_detector_model, optimizer, dataset_train_val_loader, device)
        lr_scheduler.step()
        grand_truth, preds = evaluate(object_detector_model, dataset_test_loader, device=device)
        metric.update(preds, grand_truth)
        mAP = metric.compute()

        print(
                'mAP={:>8}  mAP-50%={:>8}  mAP-75%={:>8}'.format(
                        mAP['map'].detach().item(),
                        mAP['map_50'].detach().item(),
                        mAP['map_75'].detach().item()
                )
        )
