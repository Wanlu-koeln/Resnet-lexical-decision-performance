import torch
from torch import nn, load, save, max, no_grad, randperm
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
import pandas as pd
from datetime import datetime

# Define LeNet-5 architecture
class LeNet5(nn.Module):
    def __init__(self, num_classes=2):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)  # Adjust input size based on your specific image dimensions
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the feature map
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_train_data(folder="100", N_train_samples=100, task="w_nw", base_folder="./", batch_size=64):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = ImageFolder(root=base_folder + 'task_' + task + '/' + folder + '/train', transform=transform)
    subsample_train_indices = randperm(len(train_ds))[:N_train_samples]
    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=SubsetRandomSampler(subsample_train_indices))
    return train_dl

def load_test_data(folder, task, base_folder, batch_size=64):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_ds = ImageFolder(root=base_folder + 'task_' + task + '/' + folder + '/train', transform=transform)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    return test_dl

def fit_test_model(folder=["100"], N_train_samples=[100], base_folder="./", task="w_nw", model=None, batch_size=64):
    train_dl = load_train_data(folder, N_train_samples, task, base_folder, batch_size=batch_size)
    test_dl = load_test_data(folder, task, base_folder, batch_size=batch_size)

    if model is None:
        model = LeNet5(num_classes=2)  # Create a LeNet-5 model
    else:
        # Load model weights from the previous folder if not the first folder
        if int(folder) > 100:
            prev_folder = str(int(folder) - 100)
            model.load_state_dict(load(f'lenet5_model_ari_{task}_{prev_folder}.pth'))

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for _ in range(15):
        for inputs, labels in train_dl:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Save the trained model weights for the current folder
    save(model.state_dict(), f'lenet5_model_ari_{task}_{folder}.pth')

    total = 0
    correct = 0
    model.eval()
    with no_grad():
        for inputs, labels in test_dl:
            outputs = model(inputs)
            _, predicted = max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = correct / total

    total = 0
    correct = 0
    model.eval()
    with no_grad():
        for inputs, labels in train_dl:
            outputs = model(inputs)
            _, predicted = max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    train_acc = correct / total

    return {'folder': folder, 'samples': N_train_samples, 'train': train_acc, 'test': test_acc, 'task': task}

def train_and_test_sequentially(folder_list=["100"], N_train_samples_list=[100], base_folder="./", task="w_nw", iterations=10, batch_size=64):
    model = None
    results = []

    for folder in folder_list:
        for N_train_samples in N_train_samples_list:
            print(f"Training and testing in folder '{folder}', samples = {N_train_samples}, batch size = {batch_size}")
            for _ in range(iterations):
                result = fit_test_model(folder, N_train_samples, base_folder, task, model, batch_size=batch_size)
                print(f"Model saved to LeNet5_model_ari_{task}_{folder}.pth")
                model = LeNet5(num_classes=2)  # Create a new LeNet-5 model
                model.load_state_dict(load(f'LeNet5_model_ari_{task}_{folder}.pth'))
                results.append(result)
                # Optionally, process or store the 'result' data as needed
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"lenet5_results_ari_add_model_{task}_batch_size_{batch_size}.csv", index=False)

if __name__ == "__main__":
    # Experiment with different batch sizes
    batch_sizes = [32, 64, 128]
    folder_list = [str(i) for i in range(100, 56600)]
    N_train_samples_list = [100]
    base_folder = '/Users/wanlufu/Downloads/images_baboon/Dataset_ari_steps1k/'
    task = "w_nw"
    iterations = 10

    for batch_size in batch_sizes:
        train_and_test_sequentially(folder_list, N_train_samples_list, base_folder, task, iterations, batch_size=batch_size)
