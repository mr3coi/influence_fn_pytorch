import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn, optim

import os

# Directories

DATA_DIR = "./data"
MODEL_DIR = "./model"

# Hyperparameters

NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.01

# Load & Preprocess Data

def create_binary_MNIST(data_dir, class1=1, class2=7):
    """
    Returns MNIST training & test datasets modified so that
    data points corresponding to only two specified labels remain.
    
    :data_dir: directory to download the MNIST data files
    :class1: the target label (newly labeled as 1)
    :class2: the label to discriminate `class1` against
                (newly labeled as 0)
    :returns: a tuple of two `torch.utils.data.Dataset` instances
                corresponding to training and test datasets
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    mnist_train = MNIST(data_dir, train=True, download=True, transform=ToTensor())
    mnist_test = MNIST(data_dir, train=False, download=True, transform=ToTensor())

    # Discard data points with labels other than `class1` nor `class2`; flatten the inputs
    # Relabel `class2` as 0 (i.e. classify `class1`s from `class2`s)
    train_indices = ((mnist_train.targets == class1) + (mnist_train.targets == class2) > 0)
    mnist_train.data = mnist_train.data[train_indices]
    mnist_train.targets = mnist_train.targets[train_indices]
    mnist_train.targets[mnist_train.targets == class2] = 0

    # Do the same with test dataset
    test_indices = ((mnist_test.targets == class1) + (mnist_test.targets == class2) > 0)
    mnist_test.data = mnist_test.data[test_indices]
    mnist_test.targets = mnist_test.targets[test_indices]
    mnist_test.targets[mnist_test.targets == class2] = 0
    
    return mnist_train, mnist_test


def preproc_binary_MNIST(inputs, targets):
    """Modifies minibatch to suit for training binary MNIST
    logistic regression classifier under BCEWithLogitsLoss.

    :inputs, targets: the minibatch to be preprocessed
    :returns: the preprocessed versions of `inputs`, `targets`
    """
    inputs = inputs.reshape(-1, 28*28)
    targets = targets.reshape(-1,1).float()
    return inputs, targets


def main():
    # Model: Logistic Regression to distinguish between 1 and 7
    # NOTE) Use `nn.BCEWithLogitsLoss` for the missing sigmoid

    logistic_reg = nn.Sequential(
        nn.Linear(784, 1),
        #nn.Sigmoid(),
    )
    mnist_train, mnist_test = create_binary_MNIST(data_dir=DATA_DIR)

    # Create DataLoaders for train & test datasets

    train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=True)

    # Train model

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(logistic_reg.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        total_train_loss = 0
        count_correct = 0
        count_total = 0
        
        for input, target in train_loader:
            optimizer.zero_grad()
            
            # Reshaping needed for `nn.BCEWithLogitsLoss`
            input, target = preproc_binary_MNIST(input, target)
            
            count_total += target.shape[0]
            
            logits = logistic_reg(input)
            loss = criterion(logits, target)
            
            count_correct += torch.sum((logits > 0).float() == target)
            
            total_train_loss += loss.item() * target.shape[0]
            loss.backward()
            optimizer.step()
        else:    # Test training progress with test dataset
            print(f"Epoch {epoch+1}) Average train loss = {total_train_loss / count_total:.4f}, "
                     f"train acccuracy = {float(count_correct) / count_total:.4f}")
            total_test_loss = 0
            count_correct = 0
            count_total = 0
            for input, target in test_loader:
                input = input.reshape(-1,28*28)
                target = target.reshape(-1,1).float()
                count_total += target.shape[0]

                logits = logistic_reg(input)
                loss = criterion(logits, target)

                count_correct += torch.sum((logits > 0).float() == target)
                total_test_loss += loss.item() * target.shape[0]
            else:
                print(f"\t\tAverage test loss = {total_test_loss / count_total:.4f}, "
                         f"test acccuracy = {float(count_correct) / count_total:.4f}")

    # Store trained model to disk

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    with open(os.path.join(MODEL_DIR, "mnist_logistic_reg.pt"), 'wb') as state_dict_f:
        torch.save(logistic_reg.state_dict(), state_dict_f)

    print(">>> Model saved to disk")


if __name__ == "__main__":
    main()
