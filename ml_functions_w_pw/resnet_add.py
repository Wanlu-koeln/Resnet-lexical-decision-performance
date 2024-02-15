from torchvision.models import resnet18
from torch import nn, load, save, max, no_grad, randperm
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
import pandas as pd
from datetime import datetime
import os

def load_train_data(folder="normal", N_train_samples=[100], task="w_pw", base_folder="./"):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = ImageFolder(root=base_folder + 'task_' + task + '/' + folder + '/train', transform=transform)
    subsample_train_indices = randperm(len(train_ds))[:N_train_samples]
    train_dl = DataLoader(train_ds, batch_size=64, sampler=SubsetRandomSampler(subsample_train_indices))
    return train_dl

def load_test_data(folder, task, base_folder):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_ds = ImageFolder(root=base_folder + 'task_' + task + '/' + folder + '/test', transform=transform)
    test_dl = DataLoader(test_ds, batch_size=64)
    return test_dl

def fit_test_model(folder=["normal"], N_train_samples=[100], base_folder="./", task="w_pw", model=None, prev_folder=""):
    train_dl = load_train_data(folder, N_train_samples, task, base_folder)
    test_dl = load_test_data(folder, task, base_folder)

    if model is None:
        model = resnet18(num_classes=2)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(63, 63), stride=(2, 2), padding=1, bias=False)
    else:
        if prev_folder and os.path.exists(f'ResNet18_model_{task}_{folder}_{prev_folder}.pth'):
            # Load model weights from the previous folder
            model.load_state_dict(load(f'ResNet18_model_{task}_{folder}_{prev_folder}.pth'))

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for _ in range(15):
        for inputs, labels in train_dl:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    if prev_folder:
        # Save the trained model weights for the current folder
        save(model.state_dict(), f'ResNet18_model_{task}_{folder}_{int(N_train_samples * 0.01)}.pth')

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


def train_and_test_sequentially(folder_list=["normal"], N_train_samples_list=[100], base_folder="./", task="w_pw", iterations=10):
    model = None
    results = []
    prev_folder = ""  # Initialize to an empty string

    for folder in folder_list:
        for N_train_samples in N_train_samples_list:
            print(f"Training and testing in folder '{folder}', samples = {N_train_samples}")
            for i in range(iterations):
                result = fit_test_model(folder, N_train_samples, base_folder, task, model, prev_folder)
                print(f"Model saved to ResNet18_model_{task}_{folder}_{N_train_samples}.pth")
                results.append(result)

                prev_folder = str(int(N_train_samples * 0.01))

                # Check if there are previous model weights to load
                if prev_folder and os.path.exists(f'ResNet18_model_{task}_{folder}_{N_train_samples}.pth'):
                    model = resnet18(num_classes=2)
                    model.conv1 = nn.Conv2d(1, 64, kernel_size=(63, 63), stride=(2, 2), padding=1, bias=False)
                    model.load_state_dict(load(f'ResNet18_model_{task}_{folder}_{N_train_samples}.pth'))
                else:
                    # If there are no previous weights, initialize the model without loading weights
                    model = resnet18(num_classes=2)
                    model.conv1 = nn.Conv2d(1, 64, kernel_size=(63, 63), stride=(2, 2), padding=1, bias=False)

            # Optionally, process or store the 'result' data as needed

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"resnet_results_add_{task}.csv", index=False)

if __name__ == "__main__":
    # Generate a list of folder names from 100 to 55400
    folder_list = ["original_oPE", "50p_oPE","normal"]  # This will create a list from "100" to "55400"
    N_train_samples_list = [round(N * 0.01),round(N * 0.02),round(N * 0.03), round(N * 0.04),round(N * 0.05),
                            round(N * 0.1),round(N * 0.2),round(N * 0.3),round(N * 0.4),round(N * 0.5),round(N * 1) ]  # Specify your sample sizes or generate programmatically
    base_folder = '/Users/wanlufu/Downloads/Dataset_oPE_24052023-4/'  # Change this to your data folder path
    task = "w_pw"  # Specify your task
    iterations = 50
    train_and_test_sequentially(folder_list, N_train_samples_list, base_folder, task, iterations)
