import os
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import libraries.neural_tangents as nt
from libraries.neural_tangents.neural_tangents import stax
import argparse


def generate_data(d, N):
    """
    Generate N uniformly distributed points on the surface of a (d-1)-dimensional unit sphere.

    Parameters:
    d (int): Dimension of the space (d-1 dimensional sphere).
    N (int): Number of points to generate.

    Returns:
    np.ndarray: An array of shape (N, d) containing the Cartesian coordinates of the points.
    """
    u = np.zeros(d)
    v = np.identity(d)

    points = np.random.multivariate_normal(mean=u, cov=v, size=N)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    S_d = points / norms

    return S_d


def create_model():
    """Creates a 4x512 ReLU MLP model."""
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(512), stax.Relu(),
        stax.Dense(512), stax.Relu(),
        stax.Dense(1)
    )
    return init_fn, apply_fn, kernel_fn


def train_model(train_data, train_labels, test_data, test_labels, lr, epochs):
    """Train the MLP model using full batch gradient descent."""
    init_fn, apply_fn, _ = create_model()
    _, params = init_fn(jax.random.PRNGKey(0), (-1, train_data.shape[1]))

    # Define the loss function
    def loss_fn(params, x, y):
        preds = apply_fn(params, x)
        return jnp.mean((preds - y) ** 2)

    # Define the gradient function
    grad_fn = jax.grad(loss_fn)

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        grads = grad_fn(params, train_data, train_labels)
        params = jax.tree_map(lambda p, g: p - lr * g, params, grads)

        train_loss = loss_fn(params, train_data, train_labels)
        test_loss = loss_fn(params, test_data, test_labels)

        train_losses.append(train_loss)
        test_losses.append(test_loss + 1)  # Adding 1 to account for noise

    return train_losses, test_losses


def experiment(train_sizes, lr, epochs, trials):
    """Run the experiment and plot the results."""
    d = 4
    test_data = generate_data(d, 1000)
    test_labels = np.ones((1000, 1))

    all_train_losses = {size: [] for size in train_sizes}
    all_test_losses = {size: [] for size in train_sizes}

    for size in train_sizes:
        for _ in range(trials):
            train_data = generate_data(d, size)
            train_labels = 1 + np.random.randn(size, 1)

            train_losses, test_losses = train_model(train_data, train_labels, test_data, test_labels, lr, epochs)
            all_train_losses[size].append(train_losses)
            all_test_losses[size].append(test_losses)

    # Average the losses over trials
    avg_train_losses = {size: np.mean(all_train_losses[size], axis=0) for size in train_sizes}
    avg_test_losses = {size: np.mean(all_test_losses[size], axis=0) for size in train_sizes}

    # Plotting the results
    plt.figure(figsize=(12, 6))
    for size in train_sizes:
        t = lr * np.arange(epochs)
        plt.plot(t, avg_train_losses[size], label=f'Train MSE (n={size})')
        plt.plot(t, avg_test_losses[size], label=f'Test MSE (n={size})')

    plt.xlabel('t (learning rate * epoch number)')
    plt.ylabel('MSE')
    plt.title('Train and Test MSE for different training set sizes')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MLP training experiment with various dataset sizes.")
    parser.add_argument("-lr", "--learning-rate", type=float, required=True, help="Learning rate for training.")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="Number of epochs for training.")
    parser.add_argument("-t", "--trials", type=int, default=5, help="Number of trials to average.")
    args = parser.parse_args()

    train_sizes = [100, 200, 500]
    experiment(train_sizes, args.learning_rate, args.epochs, args.trials)
