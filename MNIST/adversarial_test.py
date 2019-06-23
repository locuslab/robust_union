import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import sys
args = sys.argv
sys.path.append("../../")
sys.path.append("../")
# from functions import fgsm, pgd_linf, norms, norms_l1, pgd_all, pgd_l2, pgd_l1, epoch, epoch_adversarial
from adversarial_training.my_funcs import *

mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)

device = torch.device("cuda:{0}".format(str(args[1])) if torch.cuda.is_available() else "cpu")
print (device)

torch.manual_seed(0)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)    

epsilon_scheduler_inf = [0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.3,0.3,0.3]
epsilon_scheduler_l1  = [10,10,10,10,20,20,20,25,25,25]
epsilon_scheduler_l2  = [1,1,1,1,1,2,2,2,2,2]


import time
def test_model(model_name, attack, epsilon_attack, num_iter = 400, alpha = 2):
    model_test = nn.Sequential(nn.Conv2d(1, 32, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2),
#                           nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                          nn.Conv2d(32, 64, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2),
#                           nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                          Flatten(),
                          nn.Linear(7*7*64, 1024), nn.ReLU(),
                          nn.Linear(1024, 10)).to(device)
    model_test.to(device)
    model_address = model_name + ".pt"
    model_test.load_state_dict(torch.load(model_address, map_location = device))

    start = time.time()
    # if (attack == pgd_all):
    #     adv_err, adv_loss = epoch_adversarial(test_loader, model_test, attack, device = device)
    # else:
    adv_err, adv_loss = epoch(test_loader, model_test,device = device)
    # adv_err, adv_loss = epoch_adversarial(test_loader, model_test, attack, epsilon = epsilon_attack, device = device, num_iter = num_iter, alpha = alpha)
    # print ("Time for epoch: ", time.time()-start)
    print("Err: ",str(adv_err), " Loss: ", str(adv_loss))

# models_inf = ["model_cnn_linf_1", "model_cnn_linf_3_over_1", "model_cnn_linf_2_over_1"]
# models_2 = ["model_cnn_l2_1", "model_cnn_l2_2", "model_cnn_l2_3"]
# models_1 = ["model_cnn_l1_10", "model_cnn_l1_20", "model_cnn_l1_30"]
# models_arr = [models_inf, models_2, models_1]
models_arr = ["unprotected","Winners/Linf_12Jan", "Winners/17Jan_best", "Winners/L1_14Jan","8March_Exp1_extended", "8March_Exp2_linfStep0_05"]

# models_arr = ["model_cnn_schedule_milder_11Jan_extended", "model_cnn_linf_3_scheduled_11Jan_extended", 
# "model_cnn_l1_15_scheduled_12Jan_extended", "model_cnn_l2_2_scheduled_12Jan_extended"]

def frange(start, end, step):
    return [x * 0.001 for x in range(int(start*1000), int(end*1000), int(step*1000))]

# for model_list in models_arr:   
# for model in model_list:
model_name = "Winners/14March/myexp/my_exp_iter_5"
# model_list = ["9March/1_1_1_random_2_iter_0", "9March/1_1_1_random_2_iter_1", "9March/1_1_1_random_2_iter_2",
#               "9March/1_1_1_random_2_iter_3", "9March/1_1_1_random_2_iter_4", "9March/1_1_1_random_2_iter_5",
#               "9March/1_1_1_random_2_iter_6", "9March/1_1_1_random_2_iter_7", 

# model_list = ["9March/1_1_1_deterministic_iter_0"]
model_list =["9March/1_1_1_random_2_iter_0", "9March/1_1_1_random_2_iter_1", "9March/1_1_1_random_2_iter_2",
              "9March/1_1_1_random_2_iter_3", "9March/1_1_1_random_2_iter_4",
              "9March/1_1_1_random_2_iter_5", "9March/1_1_1_random_2_iter_6",
              "9March/1_1_1_random_2_iter_7" ]
model_list += ["9March/1_1_1_random_2_extend_new_iter_0", "9March/1_1_1_random_2_extend_new_iter_1", "9March/1_1_1_random_2_extend_new_iter_2",
              "9March/1_1_1_random_2_extend_new_iter_3", "9March/1_1_1_random_2_extend_new_iter_4",
              "9March/1_1_1_random_2_extend_new_iter_5", "9March/1_1_1_random_2_extend_new_iter_6",
              "9March/1_1_1_random_2_extend_new_iter_7" ]
# model_name = models_arr[2]
# model_name = "best_16Jan_extended"
# file = open("l2_refresh_13 Jan.txt", "w")
# print ("L1 attack")
epsilon = 12
alpha = 0.1
# for alpha in frange(0.05, 0.3, 0.05):
  # print ("alpha = ", alpha)
# test_model(model_name, pgd_l1, epsilon, num_iter = 200, alpha = alpha)
# for epsilon in frange (0.025, 0.425, 0.025):
#     file.write(model_name+ " "+ str(epsilon)+ "Linf Attack")
# test_model(model_name, pgd_linf, epsilon_attack = 0.3, alpha = 0.01, num_iter = 100)
# for model_name in models_arr:
#     for epsilon in frange(0.25,4.25, 0.25):
#         file.write(model_name+ " "+ str(epsilon)+ "L2 Attack")
# test_model(model_name, pgd_l2, epsilon_attack=1.5, num_iter = 100)
# for epsilon in frange(2.5,32.5,2.5):
#     print(model_name+ " "+ str(epsilon)+ "L1 Attack")
#     test_model(model_name, pgd_l1, epsilon, num_iter = 400)
model_list = ["Naive/all_out_iter_5", "Naive/triple_adv_iter_9", "9March/1_1_1_random_2_extend_new_iter_6"]
# model_list = ["9March/1_1_1_random_2_extend_new_iter_6"]
# model_list = ["April/8th/rand_0.04_alphas_[0.2, 0.25, 0.05]_iter_7", "Naive/all_out_iter_5", "Naive/triple_adv_iter_9", "9March/1_1_1_random_2_extend_new_iter_6"]
# model_list = ["April/8th/rand_0.04_alphas_[0.2, 0.25, 0.05]_iter_7"]
# model_list = ["Naive/triple_adv_iter_9"]
eps_1 = [3,6,9,12]#,20,30]#,50,60,70,80,90,100]
eps_2 = [0.5,1.0,2.0]#,5,7,10]
eps_3 = [0.1,0.2,0.3]#,0.5,0.7,1]
# model = model_list[0]
# test_model(model, pgd_l1, epsilon_attack = 12, num_iter = 100, alpha = 0.1)
# for model in model_list:
  # print(model)
  # test_model(model, pgd_l2, epsilon_attack=2.5, num_iter = 100)
  # test_model(model, pgd_linf, epsilon_attack=0.3, num_iter = 100)
# test_model(model, pgd_linf, epsilon_attack = 0.3, alpha = 0.01, num_iter = 1000)

# for model in model_list:

for model in model_list:
  # for i in range (len (eps_1)):
  # print(eps_1, " ", eps_2, " ", eps_3)
  # e_1 = eps_1[i]
  # e_2 = eps_2[i]
  # e_3 = eps_3[i]
  test_model(model, pgd_l1, epsilon_attack = 0, num_iter = 0, alpha = 0.2)
  # test_model(model, pgd_l1, epsilon_attack = e_1, num_iter = 400, alpha = 0.2)
  # test_model(model, pgd_l2, epsilon_attack=e_2, num_iter = 200)
  # test_model(model, pgd_linf, epsilon_attack = e_3, alpha = 0.01, num_iter = 200)
