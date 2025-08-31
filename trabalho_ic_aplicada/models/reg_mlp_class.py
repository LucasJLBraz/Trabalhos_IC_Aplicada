# trabalho_ic_aplicada/models/reg_mlp_class.py
import numpy as np

class MLPRegressor:
    """
    Multi-Layer Perceptron for Regression tasks.
    Uses Mean Squared Error as the loss function and backpropagation for training.
    """
    def __init__(self, hidden, activation='tanh', lr=0.01, epochs=100, l2=0.0, opt='adam', clip_grad=0.0):
        self.hidden_layers = hidden
        self.activation_name = activation
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.opt = opt
        self.clip_grad = clip_grad
        self.weights = []
        self.loss_history = []

        self._get_activation_functions()

    def _get_activation_functions(self):
        if self.activation_name == 'tanh':
            self.activation = np.tanh
            self.activation_deriv = lambda y: 1 - y**2
        elif self.activation_name == 'sigmoid':
            self.activation = lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            self.activation_deriv = lambda y: y * (1 - y)
        else: # Default to tanh
            self.activation = np.tanh
            self.activation_deriv = lambda y: 1 - y**2

    def _initialize_weights(self, X, y):
        layer_sizes = [X.shape[1]] + list(self.hidden_layers) + [y.shape[1]]
        for i in range(len(layer_sizes) - 1):
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
            W = np.random.uniform(-limit, limit, (layer_sizes[i] + 1, layer_sizes[i+1]))
            self.weights.append(W)

    def fit(self, X, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        self._initialize_weights(X, y)
        n_samples = X.shape[0]

        for epoch in range(self.epochs):
            epoch_loss = 0
            permutation = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X[permutation], y[permutation]

            for i in range(n_samples):
                xi, yi = X_shuffled[i:i+1], y_shuffled[i:i+1]

                # Forward pass
                activations = [xi]
                current_input = xi
                for W in self.weights[:-1]:
                    current_input_b = np.c_[-1, current_input]
                    z = current_input_b @ W
                    a = self.activation(z)
                    activations.append(a)
                    current_input = a
                
                # Output layer (linear activation)
                current_input_b = np.c_[-1, current_input]
                y_hat = current_input_b @ self.weights[-1]
                activations.append(y_hat)

                # Backward pass
                error = yi - y_hat
                epoch_loss += np.mean(error**2)

                # Delta for output layer (linear activation)
                delta = error

                for j in range(len(self.weights) - 1, -1, -1):
                    prev_activation = activations[j]
                    prev_activation_b = np.c_[-1, prev_activation]
                    grad = prev_activation_b.T @ delta
                    
                    if self.l2 > 0:
                        grad[1:,:] += self.l2 * self.weights[j][1:,:]

                    # Propagate error
                    if j > 0:
                        delta = (delta @ self.weights[j][1:,:].T) * self.activation_deriv(prev_activation)

                    # Update weights
                    self.weights[j] += self.lr * grad

            self.loss_history.append(epoch_loss / n_samples)
        return self

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            xi = X[i:i+1]
            current_input = xi
            for W in self.weights[:-1]:
                current_input_b = np.c_[-1, current_input]
                z = current_input_b @ W
                a = self.activation(z)
                current_input = a
            
            current_input_b = np.c_[-1, current_input]
            y_hat = current_input_b @ self.weights[-1]
            predictions.append(y_hat.flatten())
        return np.array(predictions)
