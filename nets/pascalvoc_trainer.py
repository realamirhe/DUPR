import os

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm


def get_object_detection_model(
        model_path,
        model_filename,
        num_classes,
        device,
        backbone='backbone'  # backbone_f1|backbone_f2|backbone|None
):
    checkpoint = torch.load(
            os.path.join(model_path, model_filename),
            map_location=device
    )
    if backbone:
        pretrained_weight_dict = {}
        for name, weight in checkpoint['model'].items():
            if backbone in name:
                pretrained_weight_dict[name] = weight

        model = fasterrcnn_resnet50_fpn(weights=pretrained_weight_dict)
    else:
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    for images, targets in tqdm(data_loader, 'data-loader'):
        images = list(image.to(device) for image in images)

        combined_targets = []
        for detected_objects in targets:
            combined_targets.append({
                "boxes": torch.stack([t['boxes'] for t in detected_objects]).squeeze(1),
                "labels": torch.stack([t['labels'] for t in detected_objects]),
            })

        targets = [{k: v.to(device) for k, v in t.items()} for t in combined_targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    return total_loss / data_loader.__len__()


import matplotlib.pyplot as plt
import matplotlib.patches as patches


def evaluate(model, data_loader, device, save_figures=False):
    model.eval()
    preds = []
    grand_truth = []
    with torch.no_grad():
        cnt = 0
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            combined_targets = []
            for detected_objects in targets:
                combined_targets.append({
                    "boxes": torch.stack([t['boxes'] for t in detected_objects]).squeeze(1),
                    "labels": torch.stack([t['labels'] for t in detected_objects]),
                })

            targets = [{k: v.to(device) for k, v in t.items()} for t in combined_targets]
            grand_truth.extend(targets)
            out = model(images)
            preds.extend(out)
            scores = out[0]['scores'].cpu().numpy()
            inds = scores > 0.7
            bxs = out[0]['boxes'].cpu().numpy()
            bxs = bxs[inds]
            gt = targets[0]['boxes'].cpu().numpy()
            gt = gt[0]
            img = images[0].permute(1, 2, 0).cpu().numpy()
            if save_figures:
                for box in bxs:
                    fig, ax = plt.subplots(1)
                    # Display the image
                    ax.imshow(img)
                    # Create a Rectangle patch
                    rect1 = patches.Rectangle((int(box[0]), int(box[1])), abs(box[0] - box[2]),
                                              abs(box[1] - box[3]), linewidth=3, edgecolor='r', facecolor='none')
                    ax.add_patch(rect1)
                    rect2 = patches.Rectangle((int(gt[0]), int(gt[1])), abs(gt[0] - gt[2]),
                                              abs(gt[1] - gt[3]), linewidth=3, edgecolor='g', facecolor='none')
                    ax.add_patch(rect2)
                    # Add the patch to the Axes
                    fig.savefig("{}/{}.png".format(
                            pascal_voc_conf.model_output,
                            cnt), dpi=90, bbox_inches='tight')
                    cnt = cnt + 1
        # grand_truth=torch.tensor(grand_truth,device=device)
        # preds=torch.tensor(preds,device=device)
        return grand_truth, preds
