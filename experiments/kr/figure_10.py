import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_and_preprocess_data():
    train_data = pd.read_csv('data/mnist/mnist_train.csv')
    test_data = pd.read_csv('data/mnist/mnist_test.csv')

    X_train = train_data.drop(columns=['label']).values
    y_train = train_data['label'].values
    X_test = test_data.drop(columns=['label']).values
    y_test = test_data['label'].values

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, y_train, X_test, y_test


def add_label_noise(y, noise_level):
    noisy_y = y.copy()
    n_samples = len(y)
    n_noisy = int(noise_level * n_samples)
    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
    noisy_y[noisy_indices] = np.random.randint(0, 10, n_noisy)
    return noisy_y


def train_kernel_ridge(X_train, y_train, kernel, alpha=1e-6):
    model = KernelRidge(kernel=kernel, alpha=alpha)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def experiment(noise_levels, sample_sizes, X_train, y_train, X_test, y_test, kernel_type):
    results = {size: [] for size in sample_sizes}

    for noise in noise_levels:
        noisy_y_train = add_label_noise(y_train, noise)

        for size in sample_sizes:
            indices = np.random.choice(len(X_train), size, replace=False)
            X_train_sample = X_train[indices]
            y_train_sample = noisy_y_train[indices]

            model = train_kernel_ridge(X_train_sample, y_train_sample, kernel=kernel_type)
            accuracy = evaluate_model(model, X_test, y_test)
            results[size].append(1 - accuracy)  # Record test error

    return results


def plot_results(results, noise_levels, title):
    plt.figure(figsize=(10, 6))

    for size, errors in results.items():
        plt.plot(noise_levels, errors, label=f'n={size}')

    plt.xlabel('Label Noise')
    plt.ylabel('Test Error')
    plt.title(title)
    plt.legend()
    plt.show()


def main():
    X_train, y_train, X_test, y_test = load_and_preprocess_data()

    noise_levels = np.arange(0, 0.51, 0.01)
    sample_sizes = [1000, 10000, 60000]

    kernels = {
        'rbf': "Gaussian Kernel",
        'laplacian': "Laplacian Kernel"
    }

    for kernel_type, title in kernels.items():
        results = experiment(noise_levels, sample_sizes, X_train, y_train, X_test, y_test, kernel_type)
        plot_results(results, noise_levels, title)


if __name__ == "__main__":
    main()
