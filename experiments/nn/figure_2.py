import os, sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import argparse
from typing import Tuple, List, Dict

from torch.utils.data import DataLoader, Subset, Dataset, TensorDataset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'libraries')))
import libraries
from libraries.wide_resnet.networks import Wide_ResNet

class BinaryCIFAR10(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        return img, target


def load_binary_cifar10(train_path: str, test_path: str) -> Tuple[Dataset, Dataset]:
    """
    Load binary CIFAR-10 datasets from the given paths.
    """
    train_dataset = torch.load(train_path)
    test_dataset = torch.load(test_path)
    return train_dataset, test_dataset


def add_label_noise(labels: torch.Tensor, noise_level: float) -> torch.Tensor:
    """
    Add noise to the labels by flipping a certain percentage of them.
    """
    noisy_labels = labels.clone()
    n_samples = len(labels)
    n_noisy = int(noise_level * n_samples)
    noisy_indices = random.sample(range(n_samples), n_noisy)

    for idx in noisy_indices:
        # Flip the label
        noisy_labels[idx] = 'vehicle' if labels[idx] == 'animal' else 'animal'

    return noisy_labels


def labels_to_binary(labels: torch.Tensor) -> torch.Tensor:
    """
    Convert labels to binary: 0 for animal, 1 for vehicle.
    """
    return torch.tensor([1 if label == 'vehicle' else 0 for label in labels])


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        epochs: int,
        lr_schedule: List[int]
) -> Tuple[float, float]:
    """
    Train the model and evaluate its performance on the test set.
    """
    model.train()
    for epoch in range(epochs):
        if epoch in lr_schedule:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.2

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy


def save_results(avg_results: Dict[int, Dict[float, Tuple[float, float]]], log_dir: str) -> None:
    """
    Save the average results to a log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'results.txt')
    with open(log_file, 'w') as f:
        for size in avg_results:
            for noise in avg_results[size]:
                mean, std = avg_results[size][noise]
                f.write(f'Size: {size}, Noise: {noise}, Mean: {mean}, Std: {std}\n')


def experiment(
        train_sizes: List[int],
        noise_levels: List[float],
        epochs: int,
        trials: int,
        train_path: str,
        test_path: str,
        lr: float,
        momentum: float,
        batch_size: int,
        results_dir: str
) -> None:
    """
    Conduct the experiment with various training sizes, noise levels, and trials.
    """
    train_dataset, test_dataset = load_binary_cifar10(train_path, test_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_results: Dict[int, Dict[float, List[float]]] = {size: {noise: [] for noise in noise_levels} for size in
                                                        train_sizes}

    for size in train_sizes:
        for noise in noise_levels:
            for trial in range(trials):
                subset_indices = random.sample(range(len(train_dataset)), size)
                subset = Subset(train_dataset, subset_indices)
                subset_data, subset_labels = zip(*[(item[0], item[1]) for item in subset])
                subset_data = torch.stack(subset_data)
                subset_labels = torch.tensor(subset_labels)
                noisy_labels = add_label_noise(subset_labels, noise)
                binary_labels = labels_to_binary(noisy_labels)

                train_loader = DataLoader(TensorDataset(subset_data, binary_labels), batch_size=batch_size,
                                          shuffle=True)

                model = Wide_ResNet(depth=28, num_classes=2, widen_factor=10, dropRate=0.3).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

                lr_schedule = [30, 40]
                _, accuracy = train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs,
                                          lr_schedule)

                test_error = 1 - accuracy
                all_results[size][noise].append(test_error)

    # Averaging results
    avg_results = {
        size: {noise: (np.mean(all_results[size][noise]), np.std(all_results[size][noise])) for noise in noise_levels}
        for size in train_sizes}

    # Save results
    log_dir = os.path.join(results_dir, 'logs')
    save_results(avg_results, log_dir)

    # Plotting results
    graph_dir = os.path.join(results_dir, 'graphs')
    os.makedirs(graph_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    for size in train_sizes:
        noise_levels_list = sorted(avg_results[size].keys())
        means = [avg_results[size][noise][0] for noise in noise_levels_list]
        stds = [avg_results[size][noise][1] for noise in noise_levels_list]
        plt.errorbar(noise_levels_list, means, yerr=stds, label=f'n={size}', capsize=5)

    plt.title("DNN Noise Profile (Binary CIFAR-10)")
    plt.xlabel("Label Noise")
    plt.ylabel("Test Error")
    plt.legend()
    plt.grid(True)
    plt_path = os.path.join(graph_dir, 'figure_2.png')
    plt.savefig(plt_path)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Wide ResNet experiment with binary CIFAR-10.")
    parser.add_argument("--train-path", type=str, required=True, help="Path to binary CIFAR-10 train set.")
    parser.add_argument("--test-path", type=str, required=True, help="Path to binary CIFAR-10 test set.")
    args = parser.parse_args()

    train_sizes = [5000, 10000, 25000, 40000]
    noise_levels = np.arange(0, 0.51, 0.01)
    epochs = 60
    trials = 3
    learning_rate = 0.1
    momentum = 0.9
    batch_size = 128
    results_dir = "results/fig_2"

    experiment(train_sizes, noise_levels, epochs, trials, args.train_path, args.test_path, learning_rate, momentum,
               batch_size, results_dir)
