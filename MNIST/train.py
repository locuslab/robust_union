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
parser.add_argument("-gpu_id", help="Id of GPU to be used", type=int, default = 0)
parser.add_argument("-model", help="Type of Adversarial Training: \n\t 0: l_inf \n\t 1: l_1 \n\t 2: l_2 \n\t 3: msd \n\t 4: triple \n\t 5: worst \n\t 6: vanilla", type=int, default = 3)
parser.add_argument("-batch_size", help = "Batch Size for Train Set (Default = 100)", type = int, default = 100)
parser.add_argument("-lr_schedule", help = "Choice (see code) 1 or 2", type = int, default = 1)

params = parser.parse_args()
device_id = params.gpu_id
batch_size = params.batch_size
choice = params.model
lr_choice = params.lr_schedule


mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("../../data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)


device = torch.device("cuda:{0}".format(device_id) if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)    

def net():
    return nn.Sequential(nn.Conv2d(1, 32, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(32, 64, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2), Flatten(), nn.Linear(7*7*64, 1024), nn.ReLU(), nn.Linear(1024, 10))

attack_list = [ pgd_linf ,  pgd_l1_topk,   pgd_l2 ,  msd_v0 ,  triple_adv ,  pgd_worst_dir, pgd_all_old]#TRIPLE, VANILLA DON'T HAVE A ATTACK NAME ANYTHING WORKS
attack_name = ["pgd_linf", "pgd_l1_topk", "pgd_l2", "msd_v0", "triple_adv", "pgd_worst_dir", "vanilla"]
folder_name = ["LINF", "L1", "L2", "MSD_V0", "TRIPLE", "WORST", "VANILLA"]

file = open("Final/{0}/logs.txt".format(folder_name[choice]), "w")

def myprint(a):
    print(a)
    file.write(a)
    file.write("\n")

attack = attack_list[choice]
name = attack_name[choice]
folder = folder_name[choice]
epochs = 20
print (name)
criterion = nn.CrossEntropyLoss()

#### TRAIN CODE #####

if lr_choice == 1:
    lr_schedule = lambda t: np.interp([t], [0, 3, 10, epochs], [0, 0.05, 0.001, 0.0001])[0]
else:
    lr_schedule = lambda t: np.interp([t], [0, 3, 7, 15, epochs], [0.0, 0.05, 0.1, 0.001, 0.0001])[0]
# lr_schedule = lambda t: np.interp([t], [0, 3, 7, 15], [0, 0.05, 0.001, 0.0001])[0]
model = net().to(device)
opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# model.load_state_dict(torch.load("Models/PGD_all_topk/" + name + "_iter_5.pt"))
for t in 1,range(epochs+1):
    start = time.time()
    print ("Learning Rate = ", lr_schedule(t))
    if choice == 6:
        train_loss, train_acc = epoch(train_loader, lr_schedule, model, epoch_i = t, opt = opt, device = device)
    elif choice == 4:
        train_loss, train_acc = triple_adv(train_loader, lr_schedule, model, epoch_i = t, attack = attack, opt = opt, device = device)
    else:
        train_loss, train_acc = epoch_adversarial(train_loader, lr_schedule, model, epoch_i = t, attack = attack, opt = opt, device = device)

    test_loss, test_acc = epoch(test_loader, lr_schedule,  model, epoch_i = t,  device = device, stop = True)
    linf_loss, linf_acc = epoch_adversarial(test_loader, lr_schedule,  model, epoch_i = t, attack = pgd_linf, device = device, stop = True)
    l2_loss, l2_acc = epoch_adversarial(test_loader, lr_schedule,  model, epoch_i = t, attack = pgd_l2, device = device, stop = True)
    l1_loss, l1_acc_topk = epoch_adversarial(test_loader, lr_schedule,  model, epoch_i = t, attack = pgd_l1_topk, device = device, stop = True)

    myprint(*("{:.6f}".format(i) for i in (train_acc,test_acc, linf_acc, l2_acc, l1_acc_topk)), sep="\t")
    myprint ("Time for epoch = ", time.time() - start)
    
    if epoch_i %5 == 0:
        torch.save(model.state_dict(), "Final/{0}/lr{1}_iter_{2}.pt".format(folder_name[choice], str(lr_choice), str(t)))


# model_train("pgd_all_const_eps_k_rand_alph_inf_0_01_k_limit_20_16Aug")
