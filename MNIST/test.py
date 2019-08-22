import foolbox
import foolbox.attacks as fa
import numpy as np
import torch
from matplotlib import pyplot as plt
import ipdb
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from mnist_funcs import *
import time
import argparse 

import sys
args = sys.argv
start = time.time()

# parser = argparse.ArgumentParser(description='Adversarial Training for MNIST', formatter_class=argparse.RawTextHelpFormatter)
# parser.add_argument("gpu_id", help="Id of GPU to be used", type=int)
# parser.add_argument("model", help="Type of Adversarial Training: \n\t 0: l_inf \n\t 1: l_1 \n\t 2: l_2 \n\t 3: msd \n\t 4: triple \n\t 5: worst \n\t 6: vanilla", type=int)
# parser.add_argument("batch_size", help = "Batch Size for Test Set (Default = 100)", type = int, default = 100)


# params = parser.parse_args()



mnist_test = datasets.MNIST("../../data", train=False, download=True, transform=transforms.ToTensor())
device = torch.device("cuda:{}".format(args[1]) if torch.cuda.is_available() else "cpu")
f = args[2]



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)    

def net():
    return nn.Sequential(nn.Conv2d(1, 32, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(32, 64, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2), Flatten(), nn.Linear(7*7*64, 1024), nn.ReLU(), nn.Linear(1024, 10))

def get_attack(attack, fmodel):
    args = []
    kwargs = {}
    # L0
    if attack == 'SAPA':
        metric = foolbox.distances.L0
        A = fa.SaltAndPepperNoiseAttack(fmodel, distance = metric)
    elif attack == 'PA':
        metric = foolbox.distances.L0
        A = fa.PointwiseAttack(fmodel, distance = metric)

    # L2
    elif 'IGD' in attack:
        metric = foolbox.distances.MSE
        A = fa.L2BasicIterativeAttack(fmodel, distance = metric)
        # kwargs['epsilons'] = 1.5
    elif attack == 'AGNA':
        metric = foolbox.distances.MSE
        kwargs['epsilons'] = np.linspace(0.5, 1, 50)
        A = fa.AdditiveGaussianNoiseAttack(fmodel, distance = metric)
    elif attack == 'BA':
        metric = foolbox.distances.MSE
        A = fa.BoundaryAttack(fmodel, distance = metric)
        kwargs['log_every_n_steps'] = 500001
    elif 'DeepFool' in attack:
        metric = foolbox.distances.MSE
        A = fa.DeepFoolL2Attack(fmodel, distance = metric)
    elif attack == 'PAL2':
        metric = foolbox.distances.MSE
        A = fa.PointwiseAttack(fmodel, distance = metric)

    # L inf
    elif 'FGSM' in attack and not 'IFGSM' in attack:
        metric = foolbox.distances.Linf
        A = fa.FGSM(fmodel, distance = metric)
        kwargs['epsilons'] = 20

    elif 'IFGSM' in attack:
        metric = foolbox.distances.Linf
        A = fa.IterativeGradientSignAttack(fmodel, distance = metric)
    elif 'PGD' in attack:
        metric = foolbox.distances.Linf
        A = fa.LinfinityBasicIterativeAttack(fmodel, distance = metric)
    elif 'IGM' in attack:
        metric = foolbox.distances.Linf
        A = fa.MomentumIterativeAttack(fmodel, distance = metric)
    else:
        raise Exception('Not implemented')
    return A, metric, args, kwargs



def test_foolbox(model_name, max_tests,f):
    file = open(f,"a")
    print (model_name)
    torch.manual_seed(0)
    model_test = net().to(device)
    model_address = model_name + ".pt"
    model_test.load_state_dict(torch.load(model_address, map_location = device))
    model_test.eval()
    fmodel = foolbox.models.PyTorchModel(model_test,   # return logits in shape (bs, n_classes)
                                         bounds=(0., 1.), num_classes=10,
                                         device=device)

    attacks_list = ['BA']
    # attacks_list = ['SAPA','PA','IGD','AGNA','BA','DeepFool','PAL2','FGSM','IFGSM','PGD','IGM']
    types_list   = [ 2  ]#  ,  2      , 2    , 3]
    # types_list   = [ 0    , 0  , 2   , 2    , 2  ,  2      , 2    , 3    , 3     , 3   , 3   ]
    for i in range(len(attacks_list)):
        attack_name = attacks_list[i]
        types = types_list[i]
        max_check = max_tests
        test_loader = DataLoader(mnist_test, batch_size = 1, shuffle=False)

        if attack_name == "BA":
            max_check = min(100,max_tests)
            test_loader = DataLoader(mnist_test, batch_size = 1, shuffle=False)
        start = time.time()
        file.write ("\n" + attack_name + "\n")
        attack, metric, args, kwargs = get_attack(attack_name, fmodel)
        total = 0
        err = 0
        for X,y in test_loader:
            distance = 1000
            total += 1
            image  = X[0,0,:,:].view(1,28,28).detach().numpy()
            label  = y[0].item()
            restarts = 10
            # ipdb.set_trace()
            for r in range (restarts):
                adversarial = attack(image, label=label, **kwargs)
                try :
                    adversarial.all()
                    adv = torch.from_numpy(adversarial).view(1,1,28,28).to(device)
                    adv = torch.from_numpy(adversarial)
                    # pred_label = torch.argmax(model_test(adv),dim = 1)[0]
                    distance = min(distance, norms_l2(X - adv).item())
                except:
                    a = 1
            
            file.write(str(distance) + "\n")
            # if (label != pred_label):
            #     err+=1
            # if (types == 0):
            #     file.write(str(norms_l0(X - adv).item()) + "\n")
            # elif (types == 2):
            #     file.write(str(norms(X - adv).item()) + "\n")
            # elif (types == 3):
            #     file.write(str(torch.abs(X - adv).max().item()) + "\n")
            if (total >= max_check):
                break
        print("Time Taken = ", time.time() - start)
        file.write("Time Taken = " + str(time.time() - start) + "\n")
    file.close()



def test_pgd(model_name):
    model = net().to(device)
    model_address = model_name + ".pt"
    model.load_state_dict(torch.load(model_address, map_location = device))
    attack = pgd_linf_rand
    print ("pgd_linf")
    test_loader = DataLoader(mnist_test, batch_size = 1000, shuffle=False)

    start = time.time()
    epoch_i = 0
    lr = None
    # if (attack == pgd_all):
    #     adv_err, adv_loss = epoch_adversarial(test_loader, model_test, attack, device = device)
    # else:
    # adv_loss, adv_acc = epoch(test_loader, lr, model, epoch_i, device = device)
    adv_loss, adv_acc = epoch_adversarial(test_loader, lr, model, epoch_i, attack, device = device, stop = True, num_iter = 100, restarts = 10)
    print("Acc: ",adv_acc)


# test_foolbox("Models/PGD_all_topk/pgd_all_const_eps_k_rand_alph_inf_0_01_iter_19", 1000,f)
model_list = ["l_1_iter_14", "l_2_iter_14", "l_inf_iter_14", "msd_iter_14", "naive_all_out_iter_14", "naive_triple_iter_14"]
model_list = ["msd_iter_14", "naive_all_out_iter_14", "naive_triple_iter_14"]
# choice = int(args[3])
# f = "logs/" + model_list[choice] + "_10restartsPA.txt"
# test_foolbox("Models/{0}".format(model_list[0]), 10, f)
# for i in range(11,20):
    # print (i)
# test_pgd("Models/msd_iter_14")
test_pgd("Models/PGD_all_topk/pgd_all_const_eps_k_rand_alph_inf_0_01_k_limit_20_16Aug_iter_14")

# for model in model_list:
#     print (model)
#     test_pgd("Models/{0}".format(model))
    # break

print ("Time Taken = ", time.time() - start)


eps_1 = [3,6,9,12,20,30,50,60,70,80,90,100]
eps_2 = [0.05,0.1,0.2,0.3,0.5,0.7,1,2,3,4,5,10]
eps_3 = [0.005,0.01,0.02,0.03,0.05,0.07,0.1,0.2,0.3,0.4,0.5,1]
num_1 = [50,50,50,100,100,200,200,200,300,300,300,300]
num_2 = [30,40,50,50,100,100,150,150,150,150,300,300]
num_3 = [30,40,50,50,100,100,150,150,150,150,300,300]


def test_saver(model_name):
    model = net().to(device)
    model_address = model_name + ".pt"
    model.load_state_dict(torch.load(model_address, map_location = device))
    test_batches = DataLoader(mnist_test, batch_size = 1000, shuffle=False)
    for index in range(len(eps_1)):
            e_1 = eps_1[index]
            n_1 = num_1[index]
            total_loss, total_acc_1 = epoch_adversarial_saver(test_batches, model, pgd_l1_topk, e_1, n_1, device = device)
            # print('Test Acc 1: {0:.4f}'.format(total_acc_1))    
            # break
    for index in range(len(eps_2)):        
            e_2 = eps_2[index]
            n_2 = num_2[index]
            total_loss, total_acc_2 = epoch_adversarial_saver(test_batches, model, pgd_l2, e_2, n_2, device = device)

    for index in range(len(eps_3)):
            e_3 = eps_3[index]
            n_3 = num_3[index]
            total_loss, total_acc_3 = epoch_adversarial_saver(test_batches, model, pgd_linf, e_3, n_3, device = device)
            # total_loss, total_acc_0 = epoch_adversarial(test_batches, lr_schedule, model, epoch_i,  pgd_l0, criterion, opt = None, device = device, stop = False)
        # print('Test Acc Clean: {0:.4f}, Test Acc 1: {1:.4f}, Test Acc 2: {2:.4f}, Test Acc inf: {3:.4f}, Time: {4:.1f}'.format(test_acc, total_acc_1,total_acc_2, total_acc_3, time.time() - start_time))    
        # print('Test Acc 0: {0:.4f}'.format(total_acc_0))   


# test_pgd("Models/msd_iter_14")
# test_saver("Models/msd_iter_14")