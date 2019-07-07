from __future__ import absolute_import
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import time
from mnist_funcs import *

mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)    

def net():
    return nn.Sequential(nn.Conv2d(1, 32, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(32, 64, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2), Flatten(), nn.Linear(7*7*64, 1024), nn.ReLU(), nn.Linear(1024, 10))


import time

epsilon_scheduler_inf = [0.1,0.2,0.3,0.3,0.3,0.3,0.3,0.3, 0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]
epsilon_scheduler_2 = [0.5,1.0,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5]
epsilon_scheduler_1 = [5,10,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12]
# lr_list = [1e-3,1e-3,1e-3,1e-3,1e-3,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4]

def model_train_cnn_adv(name):
    criterion = nn.CrossEntropyLoss()
    epochs = 15
    lr_schedule = lambda t: np.interp([t], [0, 3, 6, epochs], [0, 0.05, 0.001, 0.0001])[0]
    # lr_schedule = lambda t: lr_list[int(t)]
    model = net().to(device)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # moedel.load_state_dict(torch.load("9March/1_1_1_random_2_iter_7.pt"))
    # opt = optim.Adam(model.parameters(), lr=1e-3)

    for t in range(epochs):
        start = time.time()
        print ("Learning Rate = ", lr_schedule(t))
        # train_loss, train_acc = epoch(train_loader, lr_schedule, model, epoch_i = t, opt = opt, device = device)
        train_loss, train_acc = epoch_adversarial(train_loader, lr_schedule, model, 
                                                epoch_i = t, attack = pgd_all, opt = opt, 
                                                device = device, 
                                                epsilon_l_1 = epsilon_scheduler_1[t],
                                                epsilon_l_2 = epsilon_scheduler_2[t],
                                                epsilon_l_inf = epsilon_scheduler_inf[t])
        # train_err, train_loss = epoch_adversarial(train_loader, model, pgd_all, opt = opt, device = device, epsilon_l_inf = 0.3, epsilon_l_1 = 30, epsilon_l_2 = 3)
        # train_err, train_loss = epoch_adversarial(train_loader, model, pgd_all, opt = opt, device = device, 
                                    # epsilon_l_inf = 0.3, epsilon_l_1 = 15, epsilon_l_2 = 2)
        test_loss, test_acc = epoch(test_loader, lr_schedule,  model, epoch_i = t,  device = device)
        linf_loss, linf_acc = epoch_adversarial(test_loader, lr_schedule,  model, epoch_i = t, attack = pgd_linf, device = device, stop = True)
        l2_loss, l2_acc = epoch_adversarial(test_loader, lr_schedule,  model, epoch_i = t, attack = pgd_l2, device = device, stop = True)
        l1_loss, l1_acc = epoch_adversarial(test_loader, lr_schedule,  model, epoch_i = t, attack = pgd_l1, device = device, stop = True)
        print(*("{:.6f}".format(i) for i in (train_acc,test_acc, linf_acc, l2_acc, l1_acc)), sep="\t")
        print ("Time for epoch = ", time.time() - start)
        # print(*("{:.6f}".format(i) for i in (train_err, adv_err_1, adv_err_2, adv_err_inf)), sep="\t")
    torch.save(model.state_dict(), "Models/" + name + "_iter_"+ str(t)+".pt")

model_train_cnn_adv("pgd_all")