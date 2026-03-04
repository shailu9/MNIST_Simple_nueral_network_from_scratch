import numpy as np
import os

def _read_idx(fd):
    """Read an IDX file given an open file descriptor.

    The function returns the raw uint8 contents **after** the header.
    The caller is responsible for interpreting the shape.
    """

    # first 4 bytes are magic number; high byte is 0, next two are data
    # type and number of dimensions.  we only care that the magic number
    # matches the expected value for images or labels.
    magic = int.from_bytes(fd.read(4), "big")
    if magic == 0x00000803:  # images
        dims = [int.from_bytes(fd.read(4), "big") for _ in range(3)]
    elif magic == 0x00000801:  # labels
        dims = [int.from_bytes(fd.read(4), "big") for _ in range(1)]
    else:
        raise ValueError(f"Unrecognised IDX magic number: {magic}")

    # read the remainder of the file and return as uint8 array
    buf = fd.read()
    arr = np.frombuffer(buf, dtype=np.uint8)
    return arr, dims


def load_mnist_images(file_path: str) -> np.ndarray:
    """Load the MNIST images file and return a float32 array.

    Parameters
    ----------
    file_path : str
        Path to an IDX3 file containing unsigned byte image data.

    Returns
    -------
    images : np.ndarray
        Array of shape ``(num_images, rows * cols)`` with values
        scaled to ``[0.0, 1.0]``.
    """

    with open(file_path, "rb") as f:
        arr, dims = _read_idx(f)

    num, rows, cols = dims
    images = arr.reshape((num, rows * cols)).astype(np.float32) / 255.0
    return images


def load_mnist_labels(file_path: str) -> np.ndarray:
    """Load the MNIST labels file and return a 1‑D uint8 array.

    Parameters
    ----------
    file_path : str
        Path to an IDX1 file containing label data.

    Returns
    -------
    labels : np.ndarray
        Array of shape ``(num_labels,)`` containing integer labels [0..9].
    """

    with open(file_path, "rb") as f:
        arr, dims = _read_idx(f)

    (num,) = dims
    labels = arr.reshape((num,))
    return labels


def load_dataset(data_dir: str) -> tuple:
    """Convenience loader that returns train/test splits.

    Returns
    -------
    (train_images, train_labels), (test_images, test_labels)
    """

    train_images = load_mnist_images(os.path.join(data_dir, "train-images.idx3-ubyte"))
    train_labels = load_mnist_labels(os.path.join(data_dir, "train-labels.idx1-ubyte"))
    test_images = load_mnist_images(os.path.join(data_dir, "t10k-images.idx3-ubyte"))
    test_labels = load_mnist_labels(os.path.join(data_dir, "t10k-labels.idx1-ubyte"))

    return (train_images, train_labels), (test_images, test_labels)