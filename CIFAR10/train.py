import sys
sys.path.append('./models/')
import torch
from models import PreActResNet18
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
sys.path.append('./utils/')
from core import *
from torch_backend import *
from cifar_funcs import *
import ipdb
import sys 
args = sys.argv

device_id = args[1]
device = torch.device("cuda:{0}".format(device_id) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(int(device_id))

torch.cuda.device_count() 


epochs = 50
batch_size = 128
DATA_DIR = './data'
dataset = cifar10(DATA_DIR)
t = Timer()

cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255
#x -= mean*255
#x *= 1.0/(255*std)
print('Preprocessing training data')
# train_set = list(zip(transpose(normalise(pad(dataset['train']['data'], 4))), dataset['train']['labels']))
train_set = list(zip(transpose(normalise2(pad(dataset['train']['data'], 4))), dataset['train']['labels']))
print('Finished in {0:.2} seconds'.format(t()))
print('Preprocessing test data')
# test_set = list(zip(transpose(normalise(dataset['test']['data'])), dataset['test']['labels']))
test_set = list(zip(transpose(normalise2(dataset['test']['data'])), dataset['test']['labels']))
print('Finished in {0:.2} seconds'.format(t()))

train_set_x = Transform(train_set, [Crop(32, 32), FlipLR()])

train_batches = Batches(train_set_x, batch_size, shuffle=True, set_random_choices=True, num_workers=2, gpu_id = torch.cuda.current_device())
test_batches = Batches(test_set, batch_size, shuffle=False, num_workers=2, gpu_id = torch.cuda.current_device())


# model = ResNet18().cuda()
model = PreActResNet18().cuda()
for m in model.children(): 
    if not isinstance(m, nn.BatchNorm2d):
        m.half()   
        
opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
lr_schedule = lambda t: np.interp([t], [0, epochs*2//5, epochs*4//5, epochs], [0, 0.1, 0.005, 0])[0]
# model.load_state_dict(torch.load(model_name, map_location = device))
model.train()

import time

attack = pgd_l1
# print ("triple training")
# print ("L1 training")

# # for epoch_i in range(40,46,1):  
# for epoch_i in range(epochs):  
#     start_time = time.time()
#     lr = lr_schedule(epoch_i + (epoch_i+1)/len(train_batches))
# #     train_loss, train_acc = epoch(train_batches, model, epoch_i, criterion, opt, device = "cuda:1")
#     try:
#         # train_loss, train_acc = epoch_triple_adv(train_batches, lr_schedule, model, epoch_i, attack, criterion, opt, device = device)
#         train_loss, train_acc = epoch_adversarial(train_batches, lr_schedule, model, epoch_i, attack, criterion, opt = opt, device = device)
#     except:
#         # train_loss, train_acc = epoch_triple_adv(train_batches, lr_schedule, model, epoch_i, attack, criterion, opt, device = device)
#         train_loss, train_acc = epoch_adversarial(train_batches, lr_schedule, model, epoch_i, attack, criterion, opt = opt, device = device)

#     # total_loss, total_acc = epoch(test_batches, model, epoch_i, criterion, opt = None, device = "cuda:1")
# #     print (device)
#     total_loss, total_acc_1 = epoch_adversarial(test_batches, lr_schedule, model, epoch_i,  pgd_l1, criterion, opt = None, device = device, stop = True)
#     # total_loss, total_acc_2 = epoch_adversarial(test_batches, lr_schedule, model, epoch_i,  pgd_l2, criterion, opt = None, device = device, stop = True)
#     # total_loss, total_acc_3 = epoch_adversarial(test_batches, lr_schedule, model, epoch_i,  pgd_linf, criterion, opt = None, device = device, stop = True)
#     # print('Epoch: {6}, Train Acc: {5:.4f}, Test Acc 1: {4:.4f}, Test Acc 2: {3:.4f}, Test Acc inf: {2:.4f}, Time: {1:.1f}, lr: {0:.4f}'.format(lr, time.time()-start_time, total_acc_3, total_acc_2,total_acc_1,train_acc, epoch_i))    
#     print('Epoch: {4}, Train Acc: {3:.4f}, L1 Test Acc: {2:.4f}, Time: {1:.1f}, lr: {0:.4f}'.format(lr, time.time()-start_time, total_acc_1, train_acc, epoch_i))
#     if epoch_i %5 == 0:
#         # torch.save(model.state_dict(), "triple_2_iter_{0}.pt".format(str(epoch_i)))
#         torch.save(model.state_dict(), "l_1_iter_{0}.pt".format(str(epoch_i)))

# def train():
attack = pgd_all_old
print("PGD ALL_OLD")

for epoch_i in range(epochs):  
    start_time = time.time()
    lr = lr_schedule(epoch_i + (epoch_i+1)/len(train_batches))
#     train_loss, train_acc = epoch(train_batches, model, epoch_i, criterion, opt, device = "cuda:1")
    try:
        train_loss, train_acc = epoch_adversarial(train_batches, lr_schedule, model, epoch_i, attack, criterion, opt, device = device)
    except:
        train_loss, train_acc = epoch_adversarial(train_batches, lr_schedule, model, epoch_i, attack, criterion, opt, device = device)

    # total_loss, total_acc = epoch(test_batches, model, epoch_i, criterion, opt = None, device = device)
#     print (device)
    total_loss, total_acc_1 = epoch_adversarial(test_batches, lr_schedule, model, epoch_i,  pgd_l1, criterion, opt = None, device = device)
    total_loss, total_acc_2 = epoch_adversarial(test_batches, lr_schedule, model, epoch_i,  pgd_l2, criterion, opt = None, device = device)
    total_loss, total_acc_3 = epoch_adversarial(test_batches, lr_schedule, model, epoch_i,  pgd_linf, criterion, opt = None, device = device)
    # total_loss, total_acc = epoch_adversarial(test_batches, lr_schedule, model, epoch_i, attack, criterion, opt = None, device = device)
    print('Epoch: {6}, Train Acc: {5:.4f}, Test Acc 1: {4:.4f}, Test Acc 2: {3:.4f}, Test Acc inf: {2:.4f}, Time: {1:.1f}, lr: {0:.4f}'.format(lr, time.time()-start_time, total_acc_3, total_acc_2,total_acc_1,train_acc, epoch_i))    
    # print('Epoch: {6}, Train Loss: {5:.4f}, Train Acc: {4:.4f}, Test Loss: {3:.4f}, Test Acc: {2:.4f}, Time: {1:.1f}, lr: {0:.4f}'.format(lr, time.time()-start_time, total_acc, total_loss,train_acc,train_loss, epoch_i))    
    # print('Epoch: {4}, Test Loss: {3:.4f}, Test Acc: {2:.4f}, Time: {1:.1f}, lr: {0:.4f}'.format(lr, time.time()-start_time, total_acc, total_loss, epoch_i))
    if epoch_i %5 == 0:
        torch.save(model.state_dict(), "pgd_old_prev_iter_{0}.pt".format(str(epoch_i)))
# In[ ]:




# torch.save(model.state_dict(), "all_15_3_3.pt")
# torch.save(model.state_dict(), "all_out_15_3_3.pt")
# torch.save(model.state_dict(), "triple_naive_15_3_3.pt")
# torch.save(model.state_dict(), "l_1_15.pt")
# torch.save(model.state_dict(), "l_inf_0_03.pt")
# torch.save(model.state_dict(), "l_2_0_3.pt")

