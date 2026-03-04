import numpy as np
import matplotlib.pyplot as plt


def one_hot(y,num_classes = 10):
    return np.eye(num_classes)[y]


def plot_image(image: np.ndarray, label: int = 0) -> None  :
    """Helper function to plot a single image."""
    plt.imshow(image.reshape(28, 28), cmap="gray")
    plt.axis("off")
    plt.title(f"Label: {label}")
    plt.show()

