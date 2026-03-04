import os
import numpy as np
import tkinter as tk
from app import DigitApp
from data_loader import load_dataset
from model import SimpleNN
from train import train, evaluate
#from utils import plot_image

DATA_DIR = ".\dataset"

def main():
    print("Hello from simple-neural-network!")
    # load the data from the default directory (./dataset)
    (x_train, y_train), (x_test, y_test) = load_dataset(DATA_DIR)

    print(f"train images: {x_train.shape}, labels: {y_train.shape}")
    print(f"test  images: {x_test.shape}, labels: {y_test.shape}")

    # visualize a single random image from the training set
    #plot_image(x_train[0], y_train[0])
    paths_to_check = [
        "./weights/W1.npy",
        "./weights/b1.npy",
        "./weights/W2.npy",
        "./weights/b2.npy"
    ]
    model = SimpleNN(784, 128, 10)
    train_anyway = True  # Set to True to train the model even if weights exist, False to load existing weights
    if not train_anyway and any(os.path.exists(path) for path in paths_to_check):
        model.W1 = np.load("./weights/W1.npy")
        model.b1 = np.load("./weights/b1.npy")
        model.W2 = np.load("./weights/W2.npy")
        model.b2 = np.load("./weights/b2.npy")
    else:
        model = train(x_train, y_train)
        evaluate(model, x_test, y_test)

    root = tk.Tk()
    digit_app = DigitApp(model,root)
    root.mainloop()

if __name__ == "__main__":
    main()
