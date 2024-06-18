import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import FunctionTransformer
from tqdm import tqdm
from typing import List, Dict, Tuple
import os
import json
from concurrent.futures import ProcessPoolExecutor


def load_data(d: int, N: int) -> np.ndarray:
    """
    Load N points from the stored d-dimensional unit sphere data.
    """
    data_path = f"./data/synth/d{d}_N{N}.npy"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    S_d = np.load(data_path)
    return S_d


def gaussian_kernel_transformer(w: float = 1.0) -> FunctionTransformer:
    """
    Create a Gaussian kernel transformer with the given width w.
    """

    def kernel(X: np.ndarray) -> np.ndarray:
        pairwise_sq_dists = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1)
        return np.exp(-w ** 2 * pairwise_sq_dists)

    return FunctionTransformer(kernel, validate=False)


def kernel_ridge_regression(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        w: float = 1.0,
        ridge: float = 0.1
) -> np.ndarray:
    """
    Perform kernel ridge regression using the Gaussian kernel.
    """
    kernel_transformer = gaussian_kernel_transformer(w=w)
    K_train = kernel_transformer.transform(X_train)
    K_train += ridge * np.eye(K_train.shape[0])
    alpha = np.linalg.solve(K_train, y_train)

    pairwise_sq_dists = np.sum((X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]) ** 2, axis=-1)
    K_test = np.exp(-w ** 2 * pairwise_sq_dists)

    y_pred = K_test @ alpha
    return y_pred


def run_experiment(d: int, size: int, w: float) -> Tuple[int, float]:
    """
    Run a single iteration of the experiment for a given dimension and sample size.
    """
    X_train = load_data(d, size)
    y_train = np.random.normal(0, 1, size)
    X_test = load_data(d, 100)  # Test set size fixed to 100
    y_test = np.zeros(100)  # Clean labels

    y_pred = kernel_ridge_regression(X_train, y_train, X_test, w=w, ridge=0.1)
    mse = mean_squared_error(y_test, y_pred)

    return size, mse


def experiment(
        d: int,
        sample_sizes: List[int],
        num_runs: int = 100,
        w: float = 1.0
) -> Dict[int, List[float]]:
    """
    Conduct the experiment for different sample sizes and compute the mean squared error.
    """
    mse_results = {size: [] for size in sample_sizes}

    with ProcessPoolExecutor() as executor:
        futures = []
        for _ in range(num_runs):
            for size in sample_sizes:
                futures.append(executor.submit(run_experiment, d, size, w))

        for future in tqdm(futures, desc=f"Dimension {d}"):
            size, mse = future.result()
            mse_results[size].append(mse)

    return mse_results


def plot_results(
        results: Dict[int, Dict[int, List[float]]],
        sample_sizes: List[int],
        dimensions: List[int]
) -> None:
    """
    Plot the mean squared error results for different sample sizes and dimensions.
    """
    plt.figure(figsize=(10, 6))

    if not os.path.exists("results/fig_4/graphs"):
        os.makedirs("results/fig_4/graphs")
    if not os.path.exists("results/fig_4/logs"):
        os.makedirs("results/fig_4/logs")

    for d, mse_results in results.items():
        means = [np.mean(mse_results[size]) for size in sample_sizes]
        quantiles_25 = [np.quantile(mse_results[size], 0.25) for size in sample_sizes]
        quantiles_75 = [np.quantile(mse_results[size], 0.75) for size in sample_sizes]

        plt.plot(sample_sizes, means, label=f'd={d}')
        plt.fill_between(sample_sizes, quantiles_25, quantiles_75, alpha=0.2)

        # Save the results data to a JSON file
        log_file_path = os.path.join("results/fig_4/logs", f"results_d{d}.json")
        with open(log_file_path, 'w') as log_file:
            json.dump(mse_results, log_file)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sample Sizes')
    plt.ylabel('Test MSE')
    plt.title('Ridged Gaussian Kernel')
    plt.xticks([10, 100, 1000, 10000], labels=['10', '100', '1000', '10000'])
    plt.yticks([1, 0.1, 0.01, 0.001], labels=['$10^0$', '$10^{-1}$', '$10^{-2}$', '$10^{-3}$'])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    plot_file_path = os.path.join("results/fig_4/graphs", "fig_4.png")
    plt.savefig(plot_file_path)
    plt.close()


if __name__ == "__main__":
    sample_sizes = np.logspace(0.7, 4, num=50, dtype=int)
    dimensions = [5, 10, 15]

    results: Dict[int, Dict[int, List[float]]] = {}
    for d in dimensions:
        results[d] = experiment(d, sample_sizes, num_runs=100, w=1.0)

    plot_results(results, sample_sizes, dimensions)
