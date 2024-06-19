import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import FunctionTransformer
from tqdm import tqdm
from typing import List, Dict, Tuple
import os
import json
from concurrent.futures import ProcessPoolExecutor
import argparse
from scipy.spatial.distance import cdist
from sklearn.utils.extmath import randomized_svd

# Function to load data from a CSV file
def load_data(directory: str, filename: str, d: int, size: int) -> np.ndarray:
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        # If the file does not exist, generate and save the data
        data = generate_data(d, size)
        make_csv(data, directory, filename)
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    return data

# Function to generate data
def generate_data(d, N):
    """
    Generate N uniformly distributed points on the surface of a (d-1)-dimensional unit sphere.

    Parameters:
    d (int): Dimension of the space (d-1 dimensional sphere).
    N (int): Number of points to generate.

    Returns:
    np.ndarray: An array of shape (N, d) containing the Cartesian coordinates of the points.
    """
    # Generate d-dimensional zero vector
    u = np.zeros(d)
    # Generate dxd - dimensional identity matrix
    v = np.identity(d)

    np.random.seed(42)

    # Generate N points from a d-dimensional standard normal distribution
    points = np.random.multivariate_normal(mean=u, cov=v, size=N)

    # Normalize each point to lie on the unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    S_d = points / norms

    return S_d

# Function to save data to a CSV file
def make_csv(data, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Construct the full path
    filepath = os.path.join(directory, filename)

    # Save the points to a CSV file
    header = ','.join([f'x{i+1}' for i in range(data.shape[1])])
    np.savetxt(filepath, data, delimiter=",", header=header, comments="")
    print(f"Points saved to {filepath}")

# Gaussian kernel transformer with Nyström approximation
def gaussian_kernel_transformer_nystrom(w: float = 1.0, n_components: int = 100) -> FunctionTransformer:
    def kernel(X: np.ndarray) -> np.ndarray:
        # Ensure n_components does not exceed the number of samples
        n_components_actual = min(n_components, X.shape[0])
        subset = X[np.random.choice(X.shape[0], n_components_actual, replace=False)]
        pairwise_sq_dists = cdist(subset, subset, 'sqeuclidean')
        W = np.exp(-w ** 2 * pairwise_sq_dists)
        U, S, Vt = randomized_svd(W, n_components=n_components_actual)
        U = U[:, :n_components_actual]  # truncate to n_components_actual
        S = np.diag(S[:n_components_actual])
        pairwise_sq_dists = cdist(X, subset, 'sqeuclidean')
        K_nystrom = np.exp(-w ** 2 * pairwise_sq_dists)
        K_approx = K_nystrom @ np.linalg.inv(S) @ U.T
        return K_approx @ K_nystrom.T

    return FunctionTransformer(kernel, validate=False)

# Kernel ridge regression function
def kernel_ridge_regression(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        w: float = 1.0,
        ridge: float = 0.1,
        n_components: int = 100
) -> np.ndarray:
    kernel_transformer = gaussian_kernel_transformer_nystrom(w=w, n_components=n_components)
    K_train = kernel_transformer.transform(X_train)
    K_train += ridge * np.eye(K_train.shape[0])
    alpha = np.linalg.solve(K_train, y_train)
    pairwise_sq_dists = cdist(X_test, X_train, 'sqeuclidean')
    K_test = np.exp(-w ** 2 * pairwise_sq_dists)
    y_pred = K_test @ alpha
    return y_pred

# Run experiment function
def run_experiment(d: int, size: int, w: float, n_components: int) -> Tuple[int, float]:
    directory = "./data/synth"
    X_train = load_data(directory, f"d{d}_N{size}.csv", d, size)
    y_train = np.random.normal(0, 1, size)
    X_test = load_data(directory, f"d{d}_N100.csv", d, 100)
    y_test = np.zeros(100)
    y_pred = kernel_ridge_regression(X_train, y_train, X_test, w=w, ridge=0.1, n_components=n_components)
    mse = mean_squared_error(y_test, y_pred)
    return size, mse

# Experiment function
def experiment(
        d: int,
        sample_sizes: List[int],
        num_runs: int = 10,  # Reduced number of runs
        w: float = 1.0,
        n_components: int = 100  # Number of components for Nyström approximation
) -> Dict[int, List[float]]:
    mse_results = {int(size): [] for size in sample_sizes}  # Ensure keys are standard Python integers
    with ProcessPoolExecutor() as executor:
        futures = []
        for _ in range(num_runs):
            for size in sample_sizes:
                futures.append(executor.submit(run_experiment, d, size, w, n_components))
        for future in tqdm(futures, desc=f"Dimension {d}"):
            size, mse = future.result()
            mse_results[int(size)].append(mse)  # Ensure keys are standard Python integers
    return mse_results

# Save results function
def save_results(results: Dict[int, List[float]], output_dir: str, dimension: int) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file_path = os.path.join(output_dir, f"results_d{dimension}.json")
    try:
        with open(output_file_path, 'w') as output_file:
            json.dump(results, output_file)
        print(f"Results saved to {output_file_path}")
    except TypeError as e:
        print(f"Error saving results: {e}")

# Load results function with error handling
def load_results(input_dir: str, dimension: int) -> Dict[int, List[float]]:
    results = {}
    input_file_path = os.path.join(input_dir, f"results_d{dimension}.json")
    try:
        with open(input_file_path, 'r') as input_file:
            results = json.load(input_file)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading {input_file_path}: {e}")
    return results

# Plot results function
def plot_results(
        results: Dict[int, List[float]],
        sample_sizes: List[int],
        dimension: int
) -> None:
    if not results:
        print(f"No results to plot for dimension {dimension}.")
        return

    plt.figure(figsize=(10, 6))
    if not os.path.exists("results/fig_4/graphs"):
        os.makedirs("results/fig_4/graphs")

    means = [np.mean(results.get(size, [np.nan])) for size in sample_sizes]
    quantiles_25 = [np.quantile(results.get(size, [np.nan]), 0.25) for size in sample_sizes]
    quantiles_75 = [np.quantile(results.get(size, [np.nan]), 0.75) for size in sample_sizes]

    plt.plot(sample_sizes, means, label=f'd={dimension}')
    plt.fill_between(sample_sizes, quantiles_25, quantiles_75, alpha=0.2)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sample Sizes')
    plt.ylabel('Test MSE')
    plt.title(f'Ridged Gaussian Kernel for Dimension {dimension}')
    plt.xticks([10, 100, 1000, 10000], labels=['10', '100', '1000', '10000'])
    plt.yticks([1, 0.1, 0.01, 0.001], labels=['$10^0$', '$10^{-1}$', '$10^{-2}$', '$10^{-3}$'])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    plot_file_path = os.path.join("results/fig_4/graphs", f"fig_4_d{dimension}.png")
    plt.savefig(plot_file_path)
    plt.close()

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments and generate plots for kernel ridge regression.')
    parser.add_argument('--mse', action='store_true', help='Generate and store MSE results.')
    parser.add_argument('--plot', action='store_true', help='Generate plots from stored MSE results.')
    parser.add_argument('--dimension', type=int, help='Specify the dimension for generating or plotting results.')
    args = parser.parse_args()

    # Reduced sample sizes by a factor of 10
    sample_sizes = np.logspace(0.7, 3, num=50, dtype=int)
    dimensions = [5, 10, 15]
    output_dir = "outputs/figure_4"

    if args.mse:
        if args.dimension is None:
            raise ValueError("Please specify a dimension using --dimension when using --mse.")
        # Run the experiment and store the results for the specified dimension
        result = experiment(args.dimension, sample_sizes, num_runs=10, w=1.0, n_components=50)  # Reduced number of components
        save_results(result, output_dir, args.dimension)

    if args.plot:
        if args.dimension is None:
            raise ValueError("Please specify a dimension using --dimension when using --plot.")
        # Load the results and plot them for the specified dimension
        results = load_results(output_dir, args.dimension)
        plot_results(results, sample_sizes, args.dimension)
