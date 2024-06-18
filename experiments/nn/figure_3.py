import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


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


def load_and_preprocess_data():
    train_data = pd.read_csv('binary_mnist_train.csv')
    test_data = pd.read_csv('binary_mnist_test.csv')

    X_train = train_data.drop(columns=['label']).values
    y_train = train_data['label'].values
    X_test = test_data.drop(columns=['label']).values
    y_test = test_data['label'].values

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    y_train = np.where(y_train % 2 == 0, 1, -1)
    y_test = np.where(y_test % 2 == 0, 1, -1)

    return X_train, y_train, X_test, y_test


def add_label_noise(y, noise_level):
    noisy_y = y.copy()
    n_samples = len(y)
    n_noisy = int(noise_level * n_samples)
    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
    noisy_y[noisy_indices] = -noisy_y[noisy_indices]
    return noisy_y


def train_mlp(X_train, y_train, epochs, lr_decay_epoch, one_epoch=False):
    input_size = X_train.shape[1]
    hidden_size = 1024
    output_size = 1
    learning_rate = 1e-3

    model = MLP(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_epoch, gamma=0.1)

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

    for epoch in range(1 if one_epoch else epochs):
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        if loss.item() < 1e-4:
            break

    return model


def evaluate_model(model, X_test, y_test):
    test_tensor = torch.tensor(X_test, dtype=torch.float32)
    targets_tensor = torch.tensor(y_test, dtype=torch.float32)
    outputs = model(test_tensor)
    predicted = torch.sign(outputs.squeeze())
    accuracy = (predicted == targets_tensor).sum().item() / len(y_test)
    return accuracy


def experiment(noise_levels, sample_sizes, X_train, y_train, X_test, y_test, model_type):
    results = {size: [] for size in sample_sizes}

    for noise in noise_levels:
        noisy_y_train = add_label_noise(y_train, noise)

        for size in sample_sizes:
            indices = np.random.choice(len(X_train), size, replace=False)
            X_train_sample = X_train[indices]
            y_train_sample = noisy_y_train[indices]

            if model_type == 'mlp':
                model = train_mlp(X_train_sample, y_train_sample, epochs=100, lr_decay_epoch=[60, 90])
            elif model_type == 'one_epoch_mlp':
                model = train_mlp(X_train_sample, y_train_sample, epochs=1, lr_decay_epoch=[60, 90], one_epoch=True)
            elif model_type == '1nn':
                model = KNeighborsClassifier(n_neighbors=1)
                model.fit(X_train_sample, y_train_sample)
                accuracy = model.score(X_test, y_test)
            elif model_type == 'knn':
                k = int(np.log(len(X_train_sample)))
                model = KNeighborsClassifier(n_neighbors=k)
                model.fit(X_train_sample, y_train_sample)
                accuracy = model.score(X_test, y_test)

            if model_type in ['mlp', 'one_epoch_mlp']:
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

    experiments = {
        'mlp': "Overfit MLP",
        'one_epoch_mlp': "One-Epoch MLP",
        '1nn': "1-NN",
        'knn': "k-NN"
    }

    for model_type, title in experiments.items():
        results = experiment(noise_levels, sample_sizes, X_train, y_train, X_test, y_test, model_type)
        plot_results(results, noise_levels, title)


if __name__ == "__main__":
    main()
