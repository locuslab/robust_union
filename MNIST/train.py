from __future__ import absolute_import
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import time
import argparse
from mnist_funcs import *

parser = argparse.ArgumentParser(description='Adversarial Training for MNIST', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("gpu_id", help="Id of GPU to be used", type=int)
parser.add_argument("model", help="Type of Adversarial Training: \n\t 0: l_inf \n\t 1: l_1 \n\t 2: l_2 \n\t 3: msd \n\t 4: triple \n\t 5: worst \n\t 6: vanilla", type=int)
parser.add_argument("batch_size", help = "Batch Size for Train Set (Default = 100)", type = int, default = 100)


params = parser.parse_args()


mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("../../data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = params.batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)


device = torch.device("cuda:{0}".format(params.gpu_id) if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)    

def net():
    return nn.Sequential(nn.Conv2d(1, 32, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(32, 64, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2), Flatten(), nn.Linear(7*7*64, 1024), nn.ReLU(), nn.Linear(1024, 10))

attack_list = [pgd_l1_topk, pgd_all, pgd_all_old, pgd_all_out, pgd_all]#TRIPLE DOENST HAVE A ATTACK NAME ANYTHING WORKS
attack_name = ["pgd_l1_topk", "pgd_all", "pgd_all_old", "pgd_all_out", "pgd_triple"]
folder_name = ["L1_topk", "msd_topk", "msd_old_topk", "worst_topk", "triple_topk"]

def model_train_cnn_adv(name):
    print (name)
    criterion = nn.CrossEntropyLoss()
    epochs = 20
    # lr_schedule = lambda t: np.interp([t], [0, 3, 10, epochs], [0, 0.05, 0.001, 0.0001])[0]
    lr_schedule = lambda t: np.interp([t], [0, 3, 7, 15, epochs], [0.0, 0.05, 0.1, 0.001, 0.0001])[0]
    model = net().to(device)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # model.load_state_dict(torch.load("Models/PGD_all_topk/" + name + "_iter_5.pt"))
    attack = pgd_all
    for t in range(epochs):
        start = time.time()
        print ("Learning Rate = ", lr_schedule(t))
        # train_loss, train_acc = epoch(train_loader, lr_schedule, model, epoch_i = t, opt = opt, device = device)
        '''
        train_loss, train_acc = epoch_adversarial_tracker(train_loader, lr_schedule, model, 
                                                epoch_i = t, attack = attack, opt = opt, 
                                                device = device, 
                                                epsilon_l_1 = epsilon_scheduler_1[-1],
                                                epsilon_l_2 = epsilon_scheduler_2[-1],
                                                epsilon_l_inf = epsilon_scheduler_inf[-1])
        '''
        train_loss, train_acc = epoch_adversarial(train_loader, lr_schedule, model, epoch_i = t, attack = pgd_all, opt = opt, device = device, epsilon_l_inf = 0.3, epsilon_l_1 = 12, epsilon_l_2 = 1.5)
        # train_err, train_loss = epoch_adversarial(train_loader, model, pgd_all, opt = opt, device = device, 
                                    # epsilon_l_inf = 0.3, epsilon_l_1 = 15, epsilon_l_2 = 2)
        test_loss, test_acc = epoch(test_loader, lr_schedule,  model, epoch_i = t,  device = device, stop = True)
        linf_loss, linf_acc = epoch_adversarial(test_loader, lr_schedule,  model, epoch_i = t, attack = pgd_linf, device = device, stop = True)
        l2_loss, l2_acc = epoch_adversarial(test_loader, lr_schedule,  model, epoch_i = t, attack = pgd_l2, device = device, stop = True)
        l1_loss, l1_acc_topk = epoch_adversarial(test_loader, lr_schedule,  model, epoch_i = t, attack = pgd_l1_topk, device = device, stop = True)
        # l1_loss, l1_acc = epoch_adversarial(test_loader, lr_schedule,  model, epoch_i = t, attack = pgd_l1, device = device, stop = True)
        print(*("{:.6f}".format(i) for i in (train_acc,test_acc, linf_acc, l2_acc, l1_acc_topk)), sep="\t")
        print ("Time for epoch = ", time.time() - start)
        # print(*("{:.6f}".format(i) for i in (train_err, adv_err_1, adv_err_2, adv_err_inf)), sep="\t")
        torch.save(model.state_dict(), "Models/PGD_all_topk/" + name + "_iter_"+ str(t)+".pt")


# model_train_cnn_adv("pgd_all_const_eps_k_rand_alph_inf_0_01")