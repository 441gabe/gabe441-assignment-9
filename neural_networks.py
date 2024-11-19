import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        # Define layers and initialize weights
        # Initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros((1, output_dim))
    
    def activation(self, x):
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError("Unsupported activation function")
        
    def activation_derivative(self, x):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_fn == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation_fn == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, X):
        # Forward pass
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        out = self.z2  # No activation on the output layer for binary classification

        # Store activations for visualization
        self.activations = {'input': X, 'hidden': self.a1, 'output': out}
        return out

    def backward(self, X, y):
        # Compute gradients using chain rule
        m = X.shape[0]
        dz2 = self.activations['output'] - y
        dW2 = np.dot(self.activations['hidden'].T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.activation_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights with gradient descent
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        # Store gradients for visualization
        self.gradients = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    artists = []
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    # Plot hidden features
    hidden_features = mlp.activations['hidden']
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)

    # Hyperplane visualization in the hidden space
    # Plot hyperplane in the hidden space
    x_range = np.linspace(hidden_features[:, 0].min(), hidden_features[:, 0].max(), 100)
    y_range = np.linspace(hidden_features[:, 1].min(), hidden_features[:, 1].max(), 100)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    Z_grid = -(mlp.W2[0, 0] * X_grid + mlp.W2[1, 0] * Y_grid + mlp.b2[0, 0]) / mlp.W2[2, 0]
    ax_hidden.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.3, color='yellow')

    # Distorted input space transformed by the hidden layer
    # Distorted input space transformed by the hidden layer
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    grid = np.c_[xx.ravel(), yy.ravel()]
    hidden_grid = mlp.activation(np.dot(grid, mlp.W1) + mlp.b1)
    Z = mlp.activation(np.dot(hidden_grid, mlp.W2) + mlp.b2)
    Z = Z.reshape(xx.shape)
    ax_input.contourf(xx, yy, Z, levels=[-1, 0, 1], cmap='bwr', alpha=0.2)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolors='k')

    # Plot input layer decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = mlp.forward(grid)
    Z = Z.reshape(xx.shape)
    ax_input.contourf(xx, yy, Z, levels=[-1, 0, 1], cmap='bwr', alpha=0.2)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolors='k')

    # TODO: Visualize features and gradients as circles and edges 
    # Visualize features and gradients as circles and edges
    # FIXME: This visualization is not working properly
    
    max_w1 = np.max(np.abs(mlp.W1))
    max_b1 = np.max(np.abs(mlp.b1))
    max_w2 = np.max(np.abs(mlp.W2))
    max_b2 = np.max(np.abs(mlp.b2))

    max_val = max(max_w1, max_b1, max_w2, max_b2)

    for i in range(mlp.W1.shape[0]):
        for j in range(mlp.W1.shape[1]):
            circle = Circle((mlp.W1[i, j], mlp.b1[0, j]), radius=0.05, color='blue', alpha=0.5)
            ax_gradient.add_patch(circle)
            ax_gradient.plot([0, mlp.W1[i, j]], [0, mlp.b1[0, j]], 'k-', lw=2 * np.abs(mlp.gradients['dW1'][i, j]))

    for i in range(mlp.W2.shape[0]):
        for j in range(mlp.W2.shape[1]):
            circle = Circle((mlp.W2[i, j], mlp.b2[0, j]), radius=0.05, color='red', alpha=0.5)
            ax_gradient.add_patch(circle)
            ax_gradient.plot([0, mlp.W2[i, j]], [0, mlp.b2[0, j]], 'k-', lw=2 * np.abs(mlp.gradients['dW2'][i, j]))

    # Adjust the limits and aspect ratio to match the example image
    ax_gradient.set_xlim(-max_val, max_val)
    ax_gradient.set_ylim(-max_val, max_val)
    ax_gradient.set_aspect('equal')

    # Add grid and labels for better visualization
    ax_gradient.grid(True)
    ax_gradient.set_xlabel('Weight')
    ax_gradient.set_title('Gradient Visualization')
    
    artists.extend(ax_hidden.collections)
    artists.extend(ax_input.collections)
    artists.extend(ax_gradient.patches)
    artists.extend(ax_gradient.lines)
    return artists


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)