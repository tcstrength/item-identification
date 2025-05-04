import textwrap
from typing import List, Tuple
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import patches as patches

# def draw_boxes(image, boxes: list):
#     fig, ax = plt.subplots(1, figsize=(8, 6))
#     ax.imshow(image)
#     for box in boxes:
#         x_min, y_min, x_max, y_max = box
#         start = (x_min, y_min)
#         width = x_max - x_min
#         height = y_max - y_min
#         rect = patches.Rectangle(
#             xy=start,
#             width=width,
#             height=height,
#             linewidth=2,
#             edgecolor='r',
#             facecolor='none'
#         )
#         ax.add_patch(rect)
#     plt.show()

def crop_image(image: Image.Image, boxes: List[Tuple[int]]):
        sub_images = []
        for box in boxes:
            box = [int(coord) for coord in box]
            sub_img = image.crop(box)
            sub_images.append(sub_img)
        return sub_images

def plot_image(image):
     plt.imshow(image)
     plt.tight_layout()
     plt.show()

def draw_boxes(image, boxes: list, labels: list = None, figsize = (12, 8), max_label_width=20):
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box
        start = (x_min, y_min)
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle(
            xy=start,
            width=width,
            height=height,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)

        # Draw label if provided
        if labels and i < len(labels):
            label = labels[i]
            wrapped_label = '\n'.join(textwrap.wrap(label, width=max_label_width))

            # Adjust Y so label doesn't go above image
            label_y = y_min - 10
            if label_y < 0:
                label_y = y_min + 10

            # label_y = label_y + randint(0, 100)

            ax.text(
                x_min,
                label_y,
                wrapped_label,
                fontsize=9,
                color='white',
                verticalalignment='top',
                bbox=dict(facecolor='red', alpha=0.7, pad=2)
            )

    plt.axis('off')
    plt.tight_layout()
    plt.show()
