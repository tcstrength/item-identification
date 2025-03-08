from matplotlib import pyplot as plt
from matplotlib import patches as patches
from hcmus.models.identification_model import ModelResult

def draw_boxes(image, model_results: list[ModelResult], threshold: float=0.5):
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(image)
    for item in model_results:
        score = item.score
        box = item.box
        start = (box.x_min, box.y_min)
        width = box.x_max - box.x_min
        height = box.y_max - box.y_min
        if score > threshold:
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