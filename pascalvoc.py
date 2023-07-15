import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import TrainingArg
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
    img, target = dataset[4321]
    plot_img_bbox(img, target)

    # NOTE: select subset of dataset for demo
    dataset_train_val = Subset(dataset, list(range(1000)))
    dataset_test = Subset(dataset, list(range(1000, 1500)))

    dataset_train_val_loader = DataLoader(
            dataset_train_val,
            # TODO: read from another config reader
            batch_size=TrainingArg.batch_size_pascalVOC_train,
            num_workers=TrainingArg.num_workers,
            collate_fn=collate_fn,
            shuffle=True)
    dataset_test_loader = DataLoader(
            dataset_test,
            # TODO: read from another config reader
            batch_size=TrainingArg.batch_size_pascalVOC_test,
            num_workers=TrainingArg.num_workers,
            collate_fn=collate_fn,
            shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TODO: add local downloader!
    if(TrainingArg.is_colab):
        dupr_model_path=TrainingArg.path_colab_DUPR_model
    else:
        dupr_model_path=os.path.join(sys.path[0],TrainingArg.path_local_DUPR_model)
    # !gdown https://drive.google.com/u/0/uc?id=1-c8ZJbhMX0w5FQR5-lshwK9yZAbhUGJm&export=download
    object_detector_model = get_object_detection_model(dupr_model_path,TrainingArg.dupr_model_file_name ,num_classes=len(PascalVoc.CLASSES_NAMES))
    object_detector_model.to(device)
    print("Faster Rcnn resnet-50 has", sum(p.numel() for p in object_detector_model.parameters() if p.requires_grad),
          "parameters.")

    params = [p for p in object_detector_model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.75])
    for epoch in tqdm(range(training_config.epochs), 'epochs', leave=True):
        # training for one epoch
        loss = train_one_epoch(object_detector_model, optimizer, dataset_train_val_loader, device)

        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        grand_truth, preds = evaluate(object_detector_model, dataset_test_loader, device=device)
        metric.update(preds, grand_truth)
        mAP = metric.compute()
        print("mAP = ", mAP['map'].detach().item(),
              " map_50 = ", mAP['map_50'].detach().item(),
              " map_75 = ", mAP['map_75'].detach().item())
