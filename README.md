# Stochastic
SGD in machine learning
import numpy as np
import matplotlib.pyplot as plt

class StochasticGradientDescentRegressor:
    def __init__(self, learning_rate=0.01, epochs=50):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize weights and bias with zeros (or small random values)
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.epochs):
            total_loss = 0
            
            # Shuffle data at each epoch to ensure stochasticity
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(n_samples):
                # Select a single random sample (Stochastic step)
                xi = X_shuffled[i]
                yi = y_shuffled[i]

                # Prediction for the single sample
                y_pred = np.dot(xi, self.weights) + self.bias

                # Calculate error
                error = y_pred - yi
                
                # Compute gradients for this single sample
                dw = error * xi
                db = error

                # Update parameters immediately
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

                # Accumulate squared error for tracking loss
                total_loss += error ** 2

            # Record average loss for the epoch
            self.cost_history.append(total_loss / n_samples)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# --- Example Usage ---
if __name__ == "__main__":
    # Generate synthetic dataset
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X[:, 0] + np.random.randn(100) * 0.5

    # Initialize and train model
    sgd = StochasticGradientDescentRegressor(learning_rate=0.05, epochs=30)
    sgd.fit(X, y)

    print("Learned Weights:", sgd.weights)
    print("Learned Bias:", sgd.bias)

    # Plotting convergence (useful to include in your GitHub README)
    plt.plot(range(1, len(sgd.cost_history) + 1), sgd.cost_history, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('SGD Convergence Over Epochs')
    plt.grid(True)
    plt.show()
