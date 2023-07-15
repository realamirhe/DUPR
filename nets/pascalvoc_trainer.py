import os

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_object_detection_model(model_path,model_fileName,num_classes, device):
    ckpt_path = os.path.join(model_path, model_fileName)
    checkpoint = torch.load(ckpt_path, map_location=device)

    backbon_name = 'backbone'  # backbone_f1 backbone_f2   backbone
    pretrained_weight_dict = {}
    for name, weight in checkpoint['model'].items():
        if backbon_name in name:
            pretrained_weight_dict[name] = weight

    model = fasterrcnn_resnet50_fpn(weights=pretrained_weight_dict)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


from tqdm import tqdm


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


def evaluate(model, data_loader, device):
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
                # TODO: fix the path for colab and local
                # TODO: add the
                fig.savefig("/content/output_images/{}.png".format(cnt), dpi=90, bbox_inches='tight')
                cnt = cnt + 1
        # grand_truth=torch.tensor(grand_truth,device=device)
        # preds=torch.tensor(preds,device=device)
        return grand_truth, preds
# 80G
# drive.load
# /content/save/colab/gdrive
# /contet/
