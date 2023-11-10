from torchvision.models import squeezenet1_0
from torch import nn , save , max , no_grad , randperm
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader , SubsetRandomSampler
import torchvision.transforms as transforms
# from google.colab import drive
from numpy import mean
import pandas as pd


# drive.mount('/content/drive', force_remount=True)

# !unzip /content/drive/MyDrive/ColabNotebooks/images_for_vision_recognition_models_w_vs_cs_large.zip -d /content/drive/MyDrive/ColabNotebooks/images_for_vision_recognition_models_w_vs_cs_large_

def load_train_data ( folder = "100" , N_train_samples = 100 ) :
    transform = transforms.Compose ( [
        transforms.Grayscale () ,
        transforms.ToTensor () ,
        transforms.Normalize ( (0.5 ,) , (0.5 ,) )
    ] )
    train_ds = ImageFolder (
        root = '/Users/wanlufu/Downloads/images_baboon/Dataset_ari_steps1k/task_w_nw/' + folder + '/train' ,
        transform = transform )
    subsample_train_indices = randperm ( len ( train_ds ) )[ :N_train_samples ]
    train_dl = DataLoader ( train_ds , batch_size = 64 , sampler = SubsetRandomSampler ( subsample_train_indices ) )

    print ( "Training data loaded" )
    return train_dl


def load_test_data ( folder = "100" ) :
    transform = transforms.Compose ( [
        transforms.Grayscale () ,
        transforms.ToTensor () ,
        transforms.Normalize ( (0.5 ,) , (0.5 ,) )
    ] )
    test_ds = ImageFolder (
        root = '/Users/wanlufu/Downloads/images_baboon/Dataset_ari_steps1k/task_w_nw/' + folder + '/train' ,
        transform = transform )
    test_dl = DataLoader ( test_ds , batch_size = 64 )
    print ( "Test data loaded" )
    return test_dl


def fit_test_model ( folder = "100" , N_train_samples = 100 ) :
    train_dl = load_train_data ( folder = folder , N_train_samples = N_train_samples )
    test_dl = load_train_data ( folder = folder )

    model = squeezenet1_0 ( num_classes = 2 )
    model.features[ 0 ] = nn.Conv2d ( 1 , 96 , kernel_size = 7 , stride = 2 , padding = 3 )
    model.classifier[ 1 ] = nn.Conv2d ( 512 , 2 , kernel_size = 1 )

    # Define the optimizer and loss function
    optimizer = optim.SGD ( model.parameters () , lr = 0.1 , momentum = 0.9 )
    criterion = nn.CrossEntropyLoss ()
    # print(train_dl)

    for inputs , labels in train_dl :
        optimizer.zero_grad ()
        # print(inputs.shape)
        # print(model.conv1.weight.shape)
        outputs = model ( inputs )
        loss = criterion ( outputs , labels )
        loss.backward ()
        optimizer.step ()

    total = 0
    correct = 0
    model.eval ()
    with no_grad () :  # This is optional but saves memory
        for inputs , labels in test_dl :
            # Forward pass
            outputs = model ( inputs )
            # Calculate predictions
            _ , predicted = max ( outputs , 1 )
            # Calculate accuracy (optional)
            total += labels.size ( 0 )
            correct += (predicted == labels).sum ().item ()

    return (correct / total)


def wrapper_model_fit (
        folder_list = [ "40p" , "80p" ] , N_train_samples_list = [ 1240 , 2479 , 3719 , 4958 ] , iterations = 10
        ) :
    data = [ ]

    for folder in folder_list :
        for N_train_samples in N_train_samples_list :
            print ( str ( N_train_samples ) + " " + folder )
            acc_list = [ fit_test_model ( folder = folder , N_train_samples = N_train_samples ) for i in
                         range ( iterations ) ]
            print ( mean ( acc_list ) )
            [ data.append ( {'folder' : folder , 'train_samples' : N_train_samples , 'acc' : acc} ) for acc in
              acc_list ]

    return data