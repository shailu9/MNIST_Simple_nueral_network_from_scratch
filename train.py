import numpy as np
from numpy.random import permutation

from model import SimpleNN
from utils import one_hot


def train(x_train:np.ndarray,y_train:np.ndarray) -> SimpleNN:
    # 1. Define hyper-parameters
    input_size = 784
    hidden_size=128
    output_size=10
    y_train_one_hot = one_hot(y_train)
    epochs = 100  # Number of training iterations
    learning_rate = 0.01  # Step size for weight updates
    batch_size = 64  # Number of samples per batch

    # 2. Initialize an instance of the SimpleNN model
    model = SimpleNN(input_size, hidden_size, output_size)

    # 3. Create an empty list to store training losses
    training_losses = []

    print("Starting training...")

    # 4. Implement the training loop
    for epoch in range(epochs):
        # i. Shuffle the training data for each epoch
        permutation = np.random.permutation(x_train.shape[0])
        X_shuffled = x_train[permutation]
        y_shuffled = y_train_one_hot[permutation]

        epoch_loss = 0.0
        num_batches = 0

        # ii. Iterate through the training data in batches
        for i in range(0, X_shuffled.shape[0], batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            # iii. Perform a forward pass
            y_pred = model.forward(X_batch)

            # iv. Compute the loss for the batch
            loss = model.compute_loss(y_pred, y_batch)

            # v. Perform a backward pass to update weights and biases
            model.backward(X_batch, y_batch, learning_rate)

            # vi. Accumulate the loss for the current batch
            epoch_loss += loss
            num_batches += 1

        # b. Calculate the average loss for the epoch and append it
        avg_epoch_loss = epoch_loss / num_batches
        training_losses.append(avg_epoch_loss)

        # c. Optionally, print the epoch number and average training loss
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}")

    print("Training complete")
    model.is_trained = True
    print("Saving weights")
    np.save("./weights/W1.npy",model.W1)
    np.save("./weights/b1.npy",model.b1)
    np.save("./weights/W2.npy",model.W2)
    np.save("./weights/b2.npy",model.b2)
    return  model

def evaluate(model:SimpleNN,x_test:np.ndarray,y_test:np.ndarray)-> None:
    predictions = model.forward(x_test)
    print(f"Predictions : {predictions}")
    predicted_labels = np.argmax(predictions,axis=1)
    print(f"Predicted label {predicted_labels}")
    accuracy = np.mean(predicted_labels == y_test)
    print(f"Test Accuracy : {accuracy}")