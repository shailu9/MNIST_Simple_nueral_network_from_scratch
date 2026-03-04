# Simple Neural Network

A simple two-layer neural network implementation for digit classification using the MNIST dataset. This project includes model training, evaluation, and an interactive GUI application to test the trained model.

## Features

- **Two-Layer Neural Network**: Input layer → Hidden layer (ReLU) → Output layer (Softmax)
- **Training & Evaluation**: Includes training loop with backpropagation and model evaluation
- **Interactive GUI**: Tkinter-based application to draw digits and get predictions
- **Weight Persistence**: Trained weights are saved and reloaded for quick access
- **MNIST Dataset Support**: Automatic loading of training and test datasets

## Project Structure

```
simple-neural-network/
├── app.py              # GUI application for digit prediction
├── data_loader.py      # MNIST dataset loading utilities
├── main.py             # Main entry point
├── model.py            # Neural network model definition
├── train.py            # Training and evaluation functions
├── utils.py            # Utility functions
├── pyproject.toml      # Project configuration
├── dataset/            # MNIST dataset directory
└── weights/            # Trained model weights (NumPy format)
```

## Requirements

- Python 3.12+
- NumPy
- Pillow
- Matplotlib

## Installation

Install dependencies using:

```bash
pip install -r requirements.txt
```

Or if using the pyproject.toml:

```bash
pip install .
```

## Usage

### Running the Application

To run the interactive GUI application:

```bash
python main.py
```

The application will:
1. Load the MNIST dataset from the `dataset/` directory
2. Check for existing trained weights in the `weights/` directory
3. If weights exist, load them; otherwise, train a new model
4. Launch a GUI where you can draw digits and get predictions

### Training the Model

The model is automatically trained on first run if weights don't exist. Training parameters can be adjusted in `train.py`.

### Evaluating the Model

Model accuracy is evaluated on the test set after training. Results are printed to the console.

## Model Architecture

- **Input Layer**: 784 neurons (28×28 pixel images flattened)
- **Hidden Layer**: 128 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation (digits 0-9)

## Dataset

The MNIST dataset files should be placed in the `dataset/` directory:
- `train-images.idx3-ubyte` - Training images
- `train-labels.idx1-ubyte` - Training labels
- `t10k-images.idx3-ubyte` - Test images
- `t10k-labels.idx1-ubyte` - Test labels

## License

MIT
