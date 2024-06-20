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

# Function to load data from a CSV file
def load_data(directory: str, filename: str, d: int, size: int) -> np.ndarray:
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        # If the file does not exist, generate and save the data
        data = generate_data(d, size)
        make_csv(data, directory, filename)
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    print(f"Loaded data from {filepath}: {data.shape}")  # Debug statement
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
    print(f"Points saved to {filepath}")  # Debug statement

# Gaussian kernel function
def gaussian_kernel(X: np.ndarray, Y: np.ndarray, w: float = 1.0) -> np.ndarray:
    pairwise_sq_dists = cdist(X, Y, 'sqeuclidean')
    K = np.exp(-w ** 2 * pairwise_sq_dists)
    return K

# Kernel ridge regression function
def kernel_ridge_regression(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        w: float = 1.0,
        ridge: float = 0.1
) -> np.ndarray:
    K_train = gaussian_kernel(X_train, X_train, w=w)
    K_train += ridge * np.eye(K_train.shape[0])
    alpha = np.linalg.solve(K_train, y_train)
    K_test = gaussian_kernel(X_test, X_train, w=w)
    y_pred = K_test @ alpha
    return y_pred

# Run experiment function
def run_experiment(d: int, size: int, w: float) -> Tuple[int, float]:
    directory = "./data/synth"
    X_train = load_data(directory, f"d{d}_N{size}.csv", d, size)
    y_train = np.random.normal(0, 1, size)
    X_test = load_data(directory, f"d{d}_N100.csv", d, 100)
    y_test = np.zeros(100)
    y_pred = kernel_ridge_regression(X_train, y_train, X_test, w=w, ridge=0.1)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Run experiment with size {size}: MSE = {mse}")  # Debug statement
    return size, mse

# Experiment function
def experiment(
        d: int,
        sample_sizes: List[int],
        num_runs: int = 10,  # Reduced number of runs
        w: float = 1.0
) -> Dict[int, List[float]]:
    mse_results = {int(size): [] for size in sample_sizes}  # Ensure keys are standard Python integers

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_experiment, d, size, w) for size in sample_sizes for _ in range(num_runs)]
        for future in tqdm(futures, desc=f"Dimension {d}"):
            size, mse = future.result()
            mse_results[int(size)].append(mse)

    print(f"Experiment results for dimension {d}: {mse_results}")  # Debug statement
    return mse_results

# Save results function
def save_results(results: Dict[int, List[float]], output_dir: str, dimension: int) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file_path = os.path.join(output_dir, f"results_d{dimension}.json")
    try:
        with open(output_file_path, 'w') as output_file:
            json.dump(results, output_file)
        print(f"Results saved to {output_file_path}")  # Debug statement
    except TypeError as e:
        print(f"Error saving results: {e}")

# Load results function with error handling
def load_results(input_dir: str, dimension: int) -> Dict[int, List[float]]:
    results = {}
    input_file_path = os.path.join(input_dir, f"results_d{dimension}.json")
    try:
        with open(input_file_path, 'r') as input_file:
            results = json.load(input_file)
        print(f"Loaded results from {input_file_path}: {results}")  # Debug statement
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

    means = [np.mean(results[size]) for size in sample_sizes if size in results]

    plt.plot(sample_sizes, means, label=f'd={dimension}')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sample Sizes')
    plt.ylabel('Test MSE')
    plt.title(f'Ridged Gaussian Kernel for Dimension {dimension}')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    if not os.path.exists("results/fig_4/graphs"):
        os.makedirs("results/fig_4/graphs")
    plot_file_path = os.path.join("results/fig_4/graphs", f"fig_4_d{dimension}.png")
    plt.savefig(plot_file_path)
    plt.close()
    print(f"Plot saved to {plot_file_path}")  # Debug statement


# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments and generate plots for kernel ridge regression.')
    parser.add_argument('--mse', action='store_true', help='Generate and store MSE results.')
    parser.add_argument('--plot', action='store_true', help='Generate plots from stored MSE results.')
    parser.add_argument('--dimension', type=int, help='Specify the dimension for generating or plotting results.')
    args = parser.parse_args()

    print(f"Arguments: {args}")  # Debug statement

    sample_sizes = np.logspace(0.7, 3, num=50, dtype=int)
    dimensions = [5, 10, 15]
    output_dir = "results/fig_4/metrics"

    if args.mse:
        if args.dimension is None:
            raise ValueError("Please specify a dimension using --dimension when using --mse.")
        # Run the experiment and store the results for the specified dimension
        result = experiment(args.dimension, sample_sizes, num_runs=10, w=1.0)
        save_results(result, output_dir, args.dimension)

    if args.plot:
        if args.dimension is None:
            raise ValueError("Please specify a dimension using --dimension when using --plot.")
        # Load the results and plot them for the specified dimension
        results = load_results(output_dir, args.dimension)
        plot_results(results, sample_sizes, args.dimension)
