import math 
import matplotlib.pyplot as plt


def display_images(images, labels, images_in_row:int=5, save_path:str=None):
    if labels is not None and images.shape[0] != labels.shape[0]:
        print("labels and images with different sizes")
        labels = None

    figure = plt.figure(figsize=(8, 8))
    cols, rows = images_in_row, int(math.ceil(images.shape[0]/ images_in_row))
    for i in range(1, cols * rows + 1):
        img = images[i-1]
        figure.add_subplot(rows, cols, i)
        if labels is not None:
            plt.title(labels[i-1])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()