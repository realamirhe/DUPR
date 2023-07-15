import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image


def plot_img_bbox(img, targets):
    fig, a = plt.subplots()
    fig.set_size_inches(5, 5)
    a.imshow(to_pil_image(img))
    for t in targets:
        box = t['boxes'][0].tolist()
        x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')
        a.add_patch(rect)
    plt.show()







