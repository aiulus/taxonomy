import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import argparse

from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

from libraries.wide_resnet.networks import wide_resnet


# Load binary SVHN datasets
def load_binary_svhn(train_path, test_path):
    train_dataset = torch.load(train_path)
    test_dataset = torch.load(test_path)
    return train_dataset, test_dataset


# Add label noise
def add_label_noise(labels, noise_level, num_classes=2):
    noisy_labels = labels.clone()
    n_samples = len(labels)
    n_noisy = int(noise_level * n_samples)
    noisy_indices = random.sample(range(n_samples), n_noisy)

    for idx in noisy_indices:
        # Resample label from alternative class labels, excluding the ground truth label
        new_label = random.choice([l for l in range(num_classes) if l != labels[idx]])
        noisy_labels[idx] = new_label

    return noisy_labels


# Convert labels to binary (0 for class 0, 1 for class 1)
def labels_to_binary(labels):
    return torch.tensor([1 if label == 'class1' else 0 for label in labels])


# Train the model
def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs, lr_schedule):
    model.train()
    for epoch in range(epochs):
        if epoch in lr_schedule:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

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


# Main experiment function
def experiment(train_sizes, noise_levels, epochs, trials, train_path, test_path, lr, momentum, batch_size):
    train_dataset, test_dataset = load_binary_svhn(train_path, test_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_results = {size: {noise: [] for noise in noise_levels} for size in train_sizes}

    for size in train_sizes:
        for noise in noise_levels:
            for trial in range(trials):
                subset_indices = random.sample(range(len(train_dataset)), size)
                subset = Subset(train_dataset, subset_indices)
                subset_data, subset_labels = zip(*[(item[0], item[1]) for item in subset])
                subset_data = torch.stack(subset_data)
                subset_labels = torch.tensor(subset_labels)
                noisy_labels = add_label_noise(subset_labels, noise, num_classes=2)

                train_loader = DataLoader(TensorDataset(subset_data, noisy_labels), batch_size=batch_size, shuffle=True)

                model = wide_resnet(depth=28, num_classes=2, widen_factor=10, dropRate=0.3).to(device)
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

    # Plotting results
    plt.figure(figsize=(12, 6))
    for size in train_sizes:
        noise_levels_list = sorted(avg_results[size].keys())
        means = [avg_results[size][noise][0] for noise in noise_levels_list]
        stds = [avg_results[size][noise][1] for noise in noise_levels_list]
        plt.errorbar(noise_levels_list, means, yerr=stds, label=f'n={size}', capsize=5)

    plt.title("0/1 SVHN, overfit")
    plt.xlabel("Label Noise")
    plt.ylabel("Test Error")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Wide ResNet experiment with binary SVHN.")
    parser.add_argument("--train-path", type=str, required=True, help="Path to binary SVHN train set.")
    parser.add_argument("--test-path", type=str, required=True, help="Path to binary SVHN test set.")
    args = parser.parse_args()

    train_sizes = [9000, 18000, 36000, 70000]
    noise_levels = np.arange(0, 0.51, 0.01)
    epochs = 60
    trials = 3
    learning_rate = 0.1
    momentum = 0.9
    batch_size = 128

    experiment(train_sizes, noise_levels, epochs, trials, args.train_path, args.test_path, learning_rate, momentum,
               batch_size)
