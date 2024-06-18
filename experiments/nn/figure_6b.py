import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# Function to generate synthetic data uniformly sampled from a d-dimensional unit sphere
def generate_data(d, N):
    u = np.zeros(d)
    v = np.identity(d)
    points = np.random.multivariate_normal(mean=u, cov=v, size=N)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    S_d = points / norms
    return S_d


# Function to add label noise by randomly modifying a specified percentage of labels
def add_label_noise(y, noise_level):
    noisy_y = y.copy()
    n_noisy = int(noise_level * len(y))
    noise_indices = np.random.choice(len(y), n_noisy, replace=False)
    noisy_y[noise_indices] = np.random.uniform(-1, 1, size=n_noisy)
    return noisy_y


# MLP architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Function to train the MLP
def train_mlp(X_train, y_train, X_test, y_test, sample_size):
    input_size = X_train.shape[1]
    hidden_size = 1024
    output_size = 1
    learning_rate = 0.1
    momentum = 0.9
    batch_size = 128
    epochs = 1000

    # Learning rate schedule
    if sample_size < 120000:
        lr_decay_epochs = [150, 350]
    else:
        lr_decay_epochs = [500, 750]

    model = MLP(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_epochs, gamma=0.1)

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        if loss.item() <= 1e-4:
            break

    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32)
        outputs = model(test_tensor)
        test_loss = criterion(outputs.squeeze(), torch.tensor(y_test, dtype=torch.float32)).item()

    return test_loss


# Function to run the experiment for various noise levels and sample sizes
def run_experiment(noise_levels, sample_sizes, X_test, y_test, d=9):
    results = {size: [] for size in sample_sizes}

    for noise in noise_levels:
        for size in sample_sizes:
            X_train = generate_data(d, size)
            y_train = np.zeros(size)
            noisy_y_train = add_label_noise(y_train, noise)

            losses = []
            for _ in range(5):
                test_loss = train_mlp(X_train, noisy_y_train, X_test, y_test, size)
                losses.append(test_loss)

            avg_test_loss = np.mean(losses)
            results[size].append(avg_test_loss)

    return results


# Function to plot the results
def plot_results(results, noise_levels):
    plt.figure(figsize=(12, 8))

    for size, errors in results.items():
        plt.plot(noise_levels, errors, label=f'n={size}', linestyle='--', marker='o')

    plt.xlabel('Label Noise')
    plt.ylabel('Test MSE')
    plt.title('Test MSE as a function of Label Noise for different sample sizes')
    plt.legend()
    plt.grid(True)
    plt.show()


# Main function to orchestrate the entire process
def main():
    noise_levels = np.arange(0, 0.51, 0.01)
    sample_sizes = [300, 1000, 10000, 60000, 120000, 360000]
    d = 9

    # Generate clean test data
    X_test = generate_data(d, 10000)
    y_test = np.zeros(10000)

    results = run_experiment(noise_levels, sample_sizes, X_test, y_test, d)
    plot_results(results, noise_levels)


if __name__ == "__main__":
    main()
