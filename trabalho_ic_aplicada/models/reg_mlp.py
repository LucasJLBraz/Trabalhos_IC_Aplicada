import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

# ============================================================================
# ACTIVATION FUNCTIONS AND THEIR DERIVATIVES
# ============================================================================

def tanh_activation(x):
    """Hyperbolic tangent activation function"""
    return np.tanh(x)

def tanh_derivative(y):
    """Derivative of tanh (given the output y = tanh(x))"""
    return 1 - y**2

def sigmoid_activation(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

def sigmoid_derivative(y):
    """Derivative of sigmoid (given the output y = sigmoid(x))"""
    return y * (1 - y)

def relu_activation(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def relu_derivative(y):
    """Derivative of ReLU (given the output y = relu(x))"""
    return (y > 0).astype(float)

def linear_activation(x):
    """Linear activation function"""
    return x

def linear_derivative(y):
    """Derivative of linear activation"""
    return np.ones_like(y)

def leaky_relu_activation(x, alpha=0.01):
    """Leaky ReLU activation function"""
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(y, alpha=0.01):
    """Derivative of Leaky ReLU"""
    return np.where(y > 0, 1, alpha)

# Dictionary mapping activation names to functions
ACTIVATION_FUNCTIONS = {
    'tanh': (tanh_activation, tanh_derivative),
    'sigmoid': (sigmoid_activation, sigmoid_derivative),
    'relu': (relu_activation, relu_derivative),
    'linear': (linear_activation, linear_derivative),
    'leaky_relu': (leaky_relu_activation, leaky_relu_derivative)
}

# ============================================================================
# NETWORK INITIALIZATION
# ============================================================================

def initialize_weights(layer_sizes):
    """
    Initialize weight matrices for all layers.

    Args:
        layer_sizes: List containing the number of neurons in each layer
                    [input_features, hidden1, hidden2, ..., output]

    Returns:
        List of weight matrices (including bias terms)
    """
    weights = []

    for i in range(len(layer_sizes) - 1):
        # Xavier/Glorot initialization
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i + 1]
        limit = np.sqrt(6 / (fan_in + fan_out))

        # Add 1 for bias term
        W = np.random.uniform(-limit, limit, (layer_sizes[i + 1], layer_sizes[i] + 1))
        weights.append(W)

    return weights

# ============================================================================
# FORWARD PASS
# ============================================================================

def forward_pass(x_sample, weights, hidden_activation='tanh', output_activation='linear'):
    """
    Perform forward pass through the network.

    Args:
        x_sample: Input sample (1D array)
        weights: List of weight matrices
        hidden_activation: Activation function for hidden layers
        output_activation: Activation function for output layer

    Returns:
        activations: List of activations for each layer
        z_values: List of pre-activation values
    """
    activations = [x_sample]  # Input layer
    z_values = []  # Pre-activation values

    # Get activation functions
    hidden_func, _ = ACTIVATION_FUNCTIONS[hidden_activation]
    output_func, _ = ACTIVATION_FUNCTIONS[output_activation]

    current_input = x_sample

    for i, W in enumerate(weights):
        # Add bias term
        current_input_with_bias = np.insert(current_input, 0, -1)

        # Linear combination
        z = W @ current_input_with_bias
        z_values.append(z)

        # Apply activation function
        if i == len(weights) - 1:  # Output layer
            activation = output_func(z)
        else:  # Hidden layers
            activation = hidden_func(z)

        activations.append(activation)
        current_input = activation

    return activations, z_values

# ============================================================================
# BACKWARD PASS
# ============================================================================

def backward_pass(activations, z_values, weights, target, hidden_activation='tanh', output_activation='linear'):
    """
    Perform backward pass (backpropagation).

    Args:
        activations: List of activations from forward pass
        z_values: List of pre-activation values
        weights: List of weight matrices
        target: Target output
        hidden_activation: Activation function for hidden layers
        output_activation: Activation function for output layer

    Returns:
        gradients: List of gradients for each weight matrix
        error: Output error for MSE calculation
    """
    # Get derivative functions
    _, hidden_deriv = ACTIVATION_FUNCTIONS[hidden_activation]
    _, output_deriv = ACTIVATION_FUNCTIONS[output_activation]

    gradients = [np.zeros_like(W) for W in weights]

    # Output layer error
    output = activations[-1]
    error = target - output

    # Output layer gradient (delta)
    if output_activation == 'linear':
        delta = error  # For linear activation, derivative is 1
    else:
        delta = error * output_deriv(output)

    # Backpropagate through layers
    for i in reversed(range(len(weights))):
        # Current layer activation (with bias)
        if i == 0:
            current_activation = np.insert(activations[i], 0, -1)  # Input layer
        else:
            current_activation = np.insert(activations[i], 0, -1)  # Hidden layer

        # Calculate gradient for current layer
        delta_col = delta.reshape(-1, 1)
        activation_row = current_activation.reshape(1, -1)
        gradients[i] = delta_col @ activation_row

        # Propagate error to previous layer (if not input layer)
        if i > 0:
            # Remove bias weights for backpropagation
            W_no_bias = weights[i][:, 1:]
            error_prev = W_no_bias.T @ delta

            # Apply derivative of hidden activation
            if hidden_activation == 'tanh':
                delta = error_prev * hidden_deriv(activations[i])
            elif hidden_activation == 'sigmoid':
                delta = error_prev * hidden_deriv(activations[i])
            elif hidden_activation == 'relu':
                delta = error_prev * hidden_deriv(activations[i])
            elif hidden_activation == 'leaky_relu':
                delta = error_prev * leaky_relu_derivative(activations[i])
            else:  # linear
                delta = error_prev * hidden_deriv(activations[i])

    return gradients, error

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_mlp_regression(X_train, y_train, layer_sizes, epochs, eta_i, eta_f,
                        hidden_activation='tanh', output_activation='linear', verbose=False):
    """
    Train the MLP for regression.

    Args:
        X_train: Training input data
        y_train: Training target data
        layer_sizes: List of neurons per layer [input_features, hidden1, hidden2, ..., output]
        epochs: Number of training epochs
        eta_i: Initial learning rate
        eta_f: Final learning rate
        hidden_activation: Activation function for hidden layers
        output_activation: Activation function for output layer
        verbose: Whether to print progress

    Returns:
        weights: Trained weight matrices
        curva_loss: MSE history during training
    """
    n_samples, n_features = X_train.shape
    n_outputs = y_train.shape[1] if y_train.ndim > 1 else 1

    # Ensure layer_sizes starts with input features and ends with outputs
    if layer_sizes[0] != n_features:
        layer_sizes = [n_features] + layer_sizes[1:]
    if layer_sizes[-1] != n_outputs:
        layer_sizes = layer_sizes[:-1] + [n_outputs]

    # Initialize weights
    weights = initialize_weights(layer_sizes)

    mse_history = []
    total_iterations = epochs * n_samples

    for epoch, _ in enumerate(pbar := trange(epochs, desc="Training", disable=not verbose)):
    # for epoch in range(epochs):
        # Shuffle data
        permutation = np.random.permutation(n_samples)
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        epoch_squared_error = 0.0

        for i in range(n_samples):
            x_sample = X_train_shuffled[i]
            y_target = y_train_shuffled[i]

            # Forward pass
            activations, z_values = forward_pass(x_sample, weights,
                                               hidden_activation, output_activation)

            # Backward pass
            gradients, error = backward_pass(activations, z_values, weights,
                                           y_target, hidden_activation, output_activation)

            # Accumulate squared error
            epoch_squared_error += 0.5 * np.sum(error**2)

            # Update weights with learning rate decay
            current_iteration = epoch * n_samples + i
            eta = eta_i - ((eta_i - eta_f) / total_iterations) * current_iteration

            for j, grad in enumerate(gradients):
                weights[j] += eta * grad

        mse_history.append(epoch_squared_error / n_samples)
        # if verbose:
        #     print(f"Época: {epoch + 1}/{epochs}, MSE: {curva_loss[-1]:.6f}")

        if verbose:
            pbar.set_description(f"Epoch {epoch + 1}/{epochs} | MSE: {mse_history[-1]:.6f}")

    return weights, mse_history

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_mlp_regression(X_test, weights, hidden_activation='tanh', output_activation='linear'):
    """
    Make predictions using trained MLP.

    Args:
        X_test: Test input data
        weights: Trained weight matrices
        hidden_activation: Activation function for hidden layers
        output_activation: Activation function for output layer

    Returns:
        predictions: Array of predictions
    """
    n_samples = X_test.shape[0]
    predictions = []

    for i in range(n_samples):
        x_sample = X_test[i]
        activations, _ = forward_pass(x_sample, weights, hidden_activation, output_activation)
        output = activations[-1]
        predictions.append(output)

    return np.array(predictions).flatten()
#
# # ============================================================================
# # EXAMPLE USAGE
# # ============================================================================
#
# if __name__ == "__main__":
#     # Example parameters
#     LAYER_SIZES = [2, 20, 10, 1]  # Input features will be auto-detected
#     HIDDEN_ACTIVATION = 'leaky_relu'    # Options: 'tanh', 'sigmoid', 'relu', 'leaky_relu', 'linear'
#     OUTPUT_ACTIVATION = 'linear'  # Usually 'linear' for regression
#     LEARNING_RATE_INITIAL = 0.01
#     LEARNING_RATE_FINAL = 0.001
#     EPOCHS = 100
#
#     # Generate synthetic data for demonstration
#     print("Generating synthetic data for demonstration...")
#     n_points = 400
#     X_demo = np.random.rand(n_points, 2) * 5
#     y_demo = np.sin(X_demo[:, 0]) + X_demo[:, 1]**0.5 + np.random.randn(n_points) * 0.1
#     y_demo = y_demo.reshape(-1, 1)
#
#     print(f"Data shape: X={X_demo.shape}, y={y_demo.shape}")
#     print(f"Network architecture: {LAYER_SIZES}")
#     print(f"Hidden activation: {HIDDEN_ACTIVATION}")
#     print(f"Output activation: {OUTPUT_ACTIVATION}")
#     print("-" * 50)
#
#     # Train the network
#     W, curva_loss = train_mlp_regression(
#         X_demo, y_demo,
#         layer_sizes=LAYER_SIZES,
#         epochs=EPOCHS,
#         eta_i=LEARNING_RATE_INITIAL,
#         eta_f=LEARNING_RATE_FINAL,
#         hidden_activation=HIDDEN_ACTIVATION,
#         output_activation=OUTPUT_ACTIVATION,
#         verbose=True
#     )
#
#     # Make predictions
#     y_pred = predict_mlp_regression(X_demo, W,
#                                   HIDDEN_ACTIVATION, OUTPUT_ACTIVATION)
#
#     # Calculate final MSE
#     final_mse = np.mean((y_demo.flatten() - y_pred)**2)
#     correlation = np.corrcoef(y_demo.flatten(), y_pred)[0, 1]
#
#     print(f"\nFinal MSE: {final_mse:.6f}")
#     print(f"Correlation: {correlation:.6f}")
#
#     # Plot training history
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(curva_loss)
#     plt.title('MSE During Training')
#     plt.xlabel('Epoch')
#     plt.ylabel('MSE')
#     plt.yscale('log')
#
#     plt.subplot(1, 2, 2)
#     plt.scatter(y_demo.flatten(), y_pred, alpha=0.6)
#     plt.plot([y_demo.min(), y_demo.max()], [y_demo.min(), y_demo.max()], 'r--')
#     plt.xlabel('True Values')
#     plt.ylabel('Predictions')
#     plt.title('True vs Predicted')
#
#     plt.tight_layout()
#     plt.show()