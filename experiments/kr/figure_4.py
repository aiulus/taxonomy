import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import FunctionTransformer
from tqdm import tqdm
from typing import List, Dict, Tuple
import os
import json
from concurrent.futures import ProcessPoolExecutor
from scripts.data_generation import generate_data, make_csv
# Function to load data from a CSV file
def load_data(directory: str, filename: str) -> np.ndarray:
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}")
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    return data

# Gaussian kernel transformer function
def gaussian_kernel_transformer(w: float = 1.0) -> FunctionTransformer:
    def kernel(X: np.ndarray) -> np.ndarray:
        pairwise_sq_dists = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1)
        return np.exp(-w ** 2 * pairwise_sq_dists)
    return FunctionTransformer(kernel, validate=False)

# Kernel ridge regression function
def kernel_ridge_regression(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        w: float = 1.0,
        ridge: float = 0.1
) -> np.ndarray:
    kernel_transformer = gaussian_kernel_transformer(w=w)
    K_train = kernel_transformer.transform(X_train)
    K_train += ridge * np.eye(K_train.shape[0])
    alpha = np.linalg.solve(K_train, y_train)
    pairwise_sq_dists = np.sum((X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]) ** 2, axis=-1)
    K_test = np.exp(-w ** 2 * pairwise_sq_dists)
    y_pred = K_test @ alpha
    return y_pred

# Run experiment function
def run_experiment(d: int, size: int, w: float) -> Tuple[int, float]:
    directory = "./data/synth"
    X_train = load_data(directory, f"d{d}_N{size}.csv")
    y_train = np.random.normal(0, 1, size)
    X_test = load_data(directory, f"d{d}_N100.csv")
    y_test = np.zeros(100)
    y_pred = kernel_ridge_regression(X_train, y_train, X_test, w=w, ridge=0.1)
    mse = mean_squared_error(y_test, y_pred)
    return size, mse

# Experiment function
def experiment(
        d: int,
        sample_sizes: List[int],
        num_runs: int = 100,
        w: float = 1.0
) -> Dict[int, List[float]]:
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

# Plot results function
def plot_results(
        results: Dict[int, Dict[int, List[float]]],
        sample_sizes: List[int],
        dimensions: List[int]
) -> None:
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

# Main execution
if __name__ == "__main__":
    sample_sizes = np.logspace(0.7, 4, num=50, dtype=int)
    dimensions = [5, 10, 15]

    # Run the experiments and plot the results
    results: Dict[int, Dict[int, List[float]]] = {}
    for d in dimensions:
        results[d] = experiment(d, sample_sizes, num_runs=100, w=1.0)
    plot_results(results, sample_sizes, dimensions)
