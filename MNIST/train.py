from __future__ import absolute_import
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys

from my_funcs import *

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)    

import time

def model_train_cnn_adv(name, alphas, randomness, device,**kwargs):
	opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
	criterion = nn.CrossEntropyLoss()
	lr_schedule = lambda t: np.interp([t], [0, epochs*2//5, epochs*4//5, epochs], [0, 0.1, 0.025, 0])[0]
	epochs = 10
    print (name, " ", "Adversarial Training")
    model_cnn = nn.Sequential(nn.Conv2d(1, 32, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(32, 64, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2), Flatten(), nn.Linear(7*7*64, 1024), nn.ReLU(), nn.Linear(1024, 10)).to(device)

    # model_cnn.load_state_dict(torch.load("9March/1_1_1_random_2_iter_7.pt"))
    # opt = optim.Adam(model_cnn.parameters(), lr=1e-3)

    for t in range(epochs):
        train_err, train_loss = epoch(train_loader, model_cnn, epoch = t, opt = opt, device = device)
        # train_err, train_loss = epoch_adversarial(train_loader, model_cnn, pgd_all, opt = opt, device = device, epsilon_l_inf = 0.3, epsilon_l_1 = 30, epsilon_l_2 = 3)
        # train_err, train_loss = epoch_adversarial(train_loader, model_cnn, pgd_all, opt = opt, device = device, 
                                    # epsilon_l_inf = 0.3, epsilon_l_1 = 15, epsilon_l_2 = 2)
        test_err, test_loss = epoch(test_loader, model_cnn, device = device)
        print(*("{:.6f}".format(i) for i in (train_err, adv_err_1, adv_err_2, adv_err_inf)), sep="\t")
        torch.save(model_cnn.state_dict(), name + "_iter_"+ str(t)+".pt")

