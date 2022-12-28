import numpy as np
import torchvision
import pandas as pd
import random

#import data with torchvision

train_data = torchvision.datasets.MNIST(root='data/', train=True, download=True)
test_data = torchvision.datasets.MNIST(root='data/', train=False)

train_data_data = train_data.data.numpy()
train_data_targets = train_data.targets.numpy()
test_data_data = test_data.data.numpy()
test_data_targets = test_data.targets.numpy()

def onehot(X):                      
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T

x_train, x_test = train_data_data / 255.0, test_data_data / 255.0 #

x_train = x_train.reshape(-1,784)  # flatten, (60000,28,28)（60000,784）
x_test = x_test.reshape(-1,784)  # flatten, (10000,28,28)（10000,784）

y_train = onehot(train_data_targets)   #onehot
y_test = onehot(test_data_targets) #onehot