from matplotlib import pyplot as plt
from matplotlib import patches as patches

def draw_boxes(image, boxes: list):
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(image)
    for box in boxes:
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
    plt.show()
