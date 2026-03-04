import numpy as np

class SimpleNN:
  """A simple two-layer Neural Network with ReLU activation for the hidden layer
  and Softmax for the output layer, designed for classification tasks.
  It includes methods for initialization, forward propagation, loss computation,
  backward propagation, and activation functions.
  """
  is_trained : bool = False
  def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
    """
    Initializes the weights and biases of the neural network.

    Args:
      input_size (int): The number of features in the input layer.
      hidden_size (int): The number of neurons in the hidden layer.
      output_size (int): The number of neurons in the output layer (e.g., number of classes).
    """
    self.is_trained = False
    np.random.seed(42) # Set a random seed for reproducibility of weight initialization

    # Initialize weights and biases for the hidden layer
    self.W1 = self.xavier_init((input_size, hidden_size)) # Weights from input to hidden layer
    self.b1 = np.zeros((1, hidden_size)) # Biases for the hidden layer

    # Initialize weights and biases for the output layer
    self.W2 = self.xavier_init((hidden_size, output_size)) # Weights from hidden to output layer
    self.b2 = np.zeros((1, output_size)) # Biases for the output layer

  def relu(self, X: np.ndarray) -> np.ndarray:
    """
    Applies the Rectified Linear Unit (ReLU) activation function.

    Args:
      X (np.ndarray): The input array.

    Returns:
      np.ndarray: The output array with ReLU applied (max(0, X)).
    """
    return np.maximum(0, X)

  def softmax(self, X: np.ndarray) -> np.ndarray:
    """
    Applies the Softmax activation function to convert raw scores (logits)
    into probabilities that sum to 1 across each output for multi-class classification.
    This implementation includes a common numerical stability trick.

    Args:
      X (np.ndarray): The input array of raw scores (logits).

    Returns:
      np.ndarray: An array of probabilities.
    """
    # Subtract the maximum value from X for numerical stability
    # This prevents exp(X) from becoming too large and causing overflow.
    exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
    # Divide by the sum of exponentials to get probabilities
    return exp_X / np.sum(exp_X, axis=1, keepdims=True)

  def xavier_init(self, shape: tuple) -> np.ndarray:
    """
    Initializes weights using the Xavier (Glorot) initialization method.
    This helps to keep the signal flowing through the network without exploding or vanishing gradients.

    Args:
      shape (tuple): A tuple (fan_in, fan_out) representing the number of input and output units
                     for the layer.

    Returns:
      np.ndarray: An array of initialized weights.
    """
    fan_in = shape[0] # Number of input units
    fan_out = shape[1] # Number of output units

    # Calculate the limit for the uniform distribution
    limit = np.sqrt(6 / (fan_in + fan_out))
    # Initialize weights from a uniform distribution within [-limit, limit]
    return np.random.uniform(-limit, limit, shape)

  def forward(self, X: np.ndarray) -> np.ndarray:
    """
    Performs a forward pass through the neural network to make a prediction.

    Args:
      X (np.ndarray): The input data.

    Returns:
      np.ndarray: The activated output from the final layer (probabilities if softmax is used).
    """
    # Hidden layer calculation
    # Z1 = X * W1 + b1 (linear combination)
    self.Z1 = np.dot(X, self.W1) + self.b1
    # A1 = ReLU(Z1) (activation of the hidden layer)
    self.A1 = self.relu(self.Z1)

    # Output layer calculation
    # Z2 = A1 * W2 + b2 (linear combination, A1 acts as input to this layer)
    self.Z2 = np.dot(self.A1, self.W2) + self.b2
    # A2 = Softmax(Z2) (activation of the output layer for classification probabilities)
    # NOTE: Changed from relu to softmax to align with common classification tasks and compute_loss.
    self.A2 = self.softmax(self.Z2)

    return self.A2 # The final predicted probabilities

  def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Computes the categorical cross-entropy loss.

    Args:
      y_pred (np.ndarray): The predicted probabilities from the network (e.g., softmax output).
      y_true (np.ndarray): The true labels, typically in one-hot encoded format.

    Returns:
      float: The calculated average loss.
    """
    m = y_true.shape[0] # Number of samples
    # Cross-entropy loss formula. Added a small epsilon (1e-8) to log to prevent log(0) issues.
    loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
    return loss

  def relu_derivative(self, X: np.ndarray) -> np.ndarray:
    """
    Computes the derivative of the ReLU activation function.
    The derivative is 1 for positive inputs and 0 for non-positive inputs.

    Args:
      X (np.ndarray): The input array to the ReLU function.

    Returns:
      np.ndarray: An array where each element is 1 if the corresponding input was > 0, else 0.
    """
    return (X > 0).astype(float)

  def backward(self, X: np.ndarray, y_true: np.ndarray, learning_rate: float = 0.01) -> None:
    """
    Performs the backward pass (backpropagation) to compute gradients and update
    the network's weights and biases.

    Args:
      X (np.ndarray): The original input data to the network.
      y_true (np.ndarray): The true labels (one-hot encoded).
      learning_rate (float): The rate at which weights and biases are adjusted during optimization.
    """
    m = X.shape[0] # Number of samples

    # --- Output layer gradient calculation ---
    # dZ2 is the gradient of the loss with respect to Z2.
    # For softmax + cross-entropy loss, this simplifies to A2 - y_true.
    dZ2 = self.A2 - y_true
    # dW2 is the gradient of the loss with respect to W2.
    # It's the dot product of the transposed activation from the hidden layer (A1) and dZ2.
    # Transposing A1 is necessary to align dimensions for matrix multiplication (e.g., (m, hidden_size) @ (m, output_size) -> (hidden_size, output_size)).
    dW2 = np.dot(np.transpose(self.A1), dZ2) / m
    # db2 is the gradient of the loss with respect to b2.
    # It's the sum of dZ2 across samples.
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # --- Hidden layer gradient calculation ---
    # dA1 is the gradient of the loss with respect to A1.
    # It's the dot product of dZ2 and the transposed weights of the output layer (W2).
    # Transposing W2 is necessary to align dimensions for matrix multiplication (e.g., (m, output_size) @ (output_size, hidden_size) -> (m, hidden_size)).
    dA1 = np.dot(dZ2, np.transpose(self.W2))
    # dZ1 is the gradient of the loss with respect to Z1.
    # It's dA1 multiplied by the derivative of the ReLU activation for Z1.
    dZ1 = dA1 * self.relu_derivative(self.Z1)
    # dW1 is the gradient of the loss with respect to W1.
    # It's the dot product of the transposed input (X) and dZ1.
    # Transposing X is necessary to align dimensions for matrix multiplication (e.g., (m, input_size) @ (m, hidden_size) -> (input_size, hidden_size)).
    dW1 = np.dot(np.transpose(X), dZ1) / m
    # db1 is the gradient of the loss with respect to b1.
    # It's the sum of dZ1 across samples.
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # --- Adjust parameters (Weight and Bias Update) ---
    # Update W1, b1, W2, b2 using gradient descent.
    self.W1 -= learning_rate * dW1
    self.b1 -= learning_rate * db1
    self.W2 -= learning_rate * dW2
    self.b2 -= learning_rate * db2