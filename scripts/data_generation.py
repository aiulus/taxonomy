import os
import argparse

import numpy as np
import pandas as pd

import torch
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor, Lambda

import numpy as np
import os

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

def make_csv(data, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Construct the full path
    filepath = os.path.join(directory, filename)

    # Save the points to a CSV file
    header = ','.join([f'x{i+1}' for i in range(data.shape[1])])
    np.savetxt(filepath, data, delimiter=",", header=header, comments="")
    print(f"Points saved to {filepath}")

def generate_f4(output_dir):
    # Adjust sample sizes to match the downsized experiment parameters
    sample_sizes = np.logspace(0.7, 3, num=50, dtype=int)
    dimensions = [5, 10, 15]

    # Generate and save the data
    for d in dimensions:
        for size in sample_sizes:
            data = generate_data(d, size)
            make_csv(data, "./data/synth", f"d{d}_N{size}.csv")
        # Generate a fixed test set of 100 points
        data = generate_data(d, 100)
        make_csv(data, "./data/synth", f"d{d}_N100.csv")

# Execute the data generation
output_dir = "./data/synth"
generate_f4(output_dir)



def process_mnist():
    transform = Lambda(lambda x: x.view(-1).numpy())

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())

    # Extract the data and targets
    train_data = train_dataset.data.view(train_dataset.data.size(0), -1).numpy()
    train_targets = train_dataset.targets.numpy()
    test_data = test_dataset.data.view(test_dataset.data.size(0), -1).numpy()
    test_targets = test_dataset.targets.numpy()

    # Define the output directory for MNIST CSV files
    mnist_output_dir = 'data/mnist'
    if not os.path.exists(mnist_output_dir):
        os.makedirs(mnist_output_dir)

    # Save the flattened training and test sets to .csv files
    train_df = pd.DataFrame(train_data)
    train_df['label'] = train_targets
    train_df.to_csv(os.path.join(mnist_output_dir, 'mnist_train.csv'), index=False)

    test_df = pd.DataFrame(test_data)
    test_df['label'] = test_targets
    test_df.to_csv(os.path.join(mnist_output_dir, 'mnist_test.csv'), index=False)

    # Map the original targets to binary labels
    train_binary_labels = [1 if target % 2 == 0 else -1 for target in train_targets]
    test_binary_labels = [1 if target % 2 == 0 else -1 for target in test_targets]

    # Save the binary labeled datasets
    train_df['binary_label'] = train_binary_labels
    train_df.to_csv(os.path.join(mnist_output_dir, 'binary_mnist_train.csv'), index=False)

    test_df['binary_label'] = test_binary_labels
    test_df.to_csv(os.path.join(mnist_output_dir, 'binary_mnist_test.csv'), index=False)

    print("MNIST files saved successfully.")

class BinaryCIFAR10(datasets.CIFAR10):
    def __init__(self, *args, binarized_targets=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.binarized_targets = binarized_targets

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        target = self.binarized_targets[index]
        return img, target

def process_cifar():
    data_path = 'data/cifar10'
    os.makedirs(data_path, exist_ok=True)

    transform = ToTensor()
    cifar_train = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    cifar_test = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)

    vehicle_classes = [0, 1, 8, 9]  # airplanes, cars, ships, trucks
    animal_classes = [2, 3, 4, 5, 6, 7]  # birds, cats, deer, dogs, frogs, horses

    torch.save(cifar_train, os.path.join(data_path, 'cifar_train.pth'))
    torch.save(cifar_test, os.path.join(data_path, 'cifar_test.pth'))

    def binarize_targets(dataset, vehicle_classes, animal_classes):
        binarized_targets = []
        for target in dataset.targets:
            if target in vehicle_classes:
                binarized_targets.append('vehicle')
            elif target in animal_classes:
                binarized_targets.append('animal')
        return binarized_targets

    binary_cifar_train_targets = binarize_targets(cifar_train, vehicle_classes, animal_classes)
    binary_cifar_test_targets = binarize_targets(cifar_test, vehicle_classes, animal_classes)

    binary_cifar_train = BinaryCIFAR10(root=data_path, train=True, download=False, transform=transform, binarized_targets=binary_cifar_train_targets)
    binary_cifar_test = BinaryCIFAR10(root=data_path, train=False, download=False, transform=transform, binarized_targets=binary_cifar_test_targets)

    torch.save(binary_cifar_train, os.path.join(data_path, 'binary_cifar_train.pth'))
    torch.save(binary_cifar_test, os.path.join(data_path, 'binary_cifar_test.pth'))

    print("CIFAR-10 files saved successfully.")

def process_svhn():
    data_path = 'data/svhn'
    os.makedirs(data_path, exist_ok=True)

    transform = ToTensor()
    svhn_train = datasets.SVHN(root=data_path, split='train', download=True, transform=transform)
    svhn_test = datasets.SVHN(root=data_path, split='test', download=True, transform=transform)

    vehicle_classes = [0, 1, 2, 3, 4]  # 0-4 digits as vehicles
    animal_classes = [5, 6, 7, 8, 9]  # 5-9 digits as animals

    torch.save(svhn_train, os.path.join(data_path, 'svhn_train.pth'))
    torch.save(svhn_test, os.path.join(data_path, 'svhn_test.pth'))

    def binarize_targets(dataset, vehicle_classes, animal_classes):
        binarized_targets = []
        for target in dataset.labels:
            if target in vehicle_classes:
                binarized_targets.append('vehicle')
            elif target in animal_classes:
                binarized_targets.append('animal')
        return binarized_targets

    binary_svhn_train_targets = binarize_targets(svhn_train, vehicle_classes, animal_classes)
    binary_svhn_test_targets = binarize_targets(svhn_test, vehicle_classes, animal_classes)

    binary_svhn_train = datasets.SVHN(root=data_path, split='train', download=False, transform=transform)
    binary_svhn_train.labels = binary_svhn_train_targets
    binary_svhn_test = datasets.SVHN(root=data_path, split='test', download=False, transform=transform)
    binary_svhn_test.labels = binary_svhn_test_targets

    torch.save(binary_svhn_train, os.path.join(data_path, 'binary_svhn_train.pth'))
    torch.save(binary_svhn_test, os.path.join(data_path, 'binary_svhn_test.pth'))

    print("SVHN files saved successfully.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate data on the surface of a unit sphere or process datasets.")
    parser.add_argument("-s", "--synth", action="store_true", help="Flag to generate synthetic data.")
    parser.add_argument("--f4", action="store_true", help="Flag to generate synthetic data for Figure 4.")
    parser.add_argument("-m", "--mnist", action="store_true", help="Flag to process MNIST dataset.")
    parser.add_argument("-c", "--cifar", action="store_true", help="Flag to process CIFAR-10 dataset.")
    parser.add_argument("--svhn", action="store_true", help="Flag to process SVHN dataset.")
    parser.add_argument("-d", "--dimension", type=int, help="Dimension of the space (d-1 dimensional sphere). Required for synthetic data generation.")
    parser.add_argument("-N", "--points", type=int, help="Number of points to generate. Required for synthetic data generation.")
    parser.add_argument("-o", "--output-dir", type=str, default="data/synthetic",
                        help="Output directory for saving the CSV file.")

    args = parser.parse_args()

    if args.synth:
        if args.f4:
            generate_f4(args.output_dir)
        if args.dimension is None or args.points is None:
            raise ValueError("Both dimension and points must be specified for synthetic data generation.")
        # Generate data
        data = generate_data(args.dimension, args.points)
        # Construct filename with specific dimensionality and sample size
        filename = f"synt_d{args.dimension}_n{args.points}.csv"
        # Save data to CSV
        make_csv(data, args.output_dir, filename)
    elif args.mnist:
        process_mnist()
    elif args.cifar:
        process_cifar()
    elif args.svhn:
        process_svhn()
    else:
        print("Please specify either --synth, --mnist, or --cifar flag.")


