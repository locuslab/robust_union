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


# python3 test.py -gpu_id 0 -model 0 -batch_size 1 -attack 0 -restarts 10

parser = argparse.ArgumentParser(description='Adversarial Training for MNIST', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-gpu_id", help="Id of GPU to be used", type=int, default = 0)
parser.add_argument("-model", help="Type of Adversarial Training: \n\t 0: l_inf \n\t 1: l_1 \n\t 2: l_2 \n\t 3: msd \n\t 4: triple \n\t 5: worst \n\t 6: vanilla", type=int, default = 3)
parser.add_argument("-batch_size", help = "Batch Size for Test Set (Default = 100)", type = int, default = 100)
parser.add_argument("-attack", help = "Foolbox = 0; Custom PGD = 1", type = int, default = 0)
parser.add_argument("-restarts", help = "Default = 10", type = int, default = 10)


params = parser.parse_args()

device_id = params.gpu_id
batch_size = params.batch_size
choice = params.model
attack = params.attack
res = params.restarts


mnist_test = datasets.MNIST("../../data", train=False, download=True, transform=transforms.ToTensor())
device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")



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
        kwargs['log_every_n_steps'] = 5000001
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
    elif 'PGD' in attack:
        metric = foolbox.distances.Linf
        A = fa.LinfinityBasicIterativeAttack(fmodel, distance = metric)
    elif 'IGM' in attack:
        metric = foolbox.distances.Linf
        A = fa.MomentumIterativeAttack(fmodel, distance = metric)
    else:
        raise Exception('Not implemented')
    return A, metric, args, kwargs



def test_foolbox(model_name, max_tests):
    print (model_name)
    torch.manual_seed(0)
    model_test = net().to(device)
    model_address = model_name + ".pt"
    model_test.load_state_dict(torch.load(model_address, map_location = device))
    model_test.eval()
    fmodel = foolbox.models.PyTorchModel(model_test,
                                         bounds=(0., 1.), num_classes=10,
                                         device=device)

    attacks_list = ['BA']
    # attacks_list = ['SAPA','PA','IGD','AGNA','DeepFool','PAL2','FGSM','PGD','IGM']
    types_list   = [ 2  ]#  ,  2      , 2    , 3]
    # types_list   = [ 0    , 0  , 2   , 2     ,  2      , 2    , 3      , 3   , 3   ]
    norm_dict = {0:norms_l0, 1:norms_l1, 2:norms,3:norms_linf}

    for i in range(len(attacks_list)):
        file = open(model_name +"foolbox_logs.txt","a")
        restarts = res
        attack_name = attacks_list[i]
        file.write ("\n" + attack_name + "\n")
        print (attack_name )
        types = types_list[i]
        norm = norm_dict[types]
        max_check = max_tests
        test_loader = DataLoader(mnist_test, batch_size = 1, shuffle=False)

        if attack_name == "BA":
            # max_check = min(100,max_tests)
            test_loader = DataLoader(mnist_test, batch_size = 1, shuffle=False)
            restarts = 1


        start = time.time()
        output = np.ones((max_check))
        
        attack, metric, args, kwargs = get_attack(attack_name, fmodel)
        total = 0
        err = 0
        for X,y in test_loader:
            distance = 1000
            image  = X[0,0,:,:].view(1,28,28).detach().numpy()
            label  = y[0].item()
            # ipdb.set_trace()
            for r in range (restarts):
                adversarial = attack(image, label=label, **kwargs)
                try :
                    adversarial.all()
                    adv = torch.from_numpy(adversarial).view(1,1,28,28).to(device)
                    adv = torch.from_numpy(adversarial)
                    # pred_label = torch.argmax(model_test(adv),dim = 1)[0]
                    distance = min(distance, norm(X - adv).item())
                except:
                    a = 1
            
            file.write(str(distance) + "\n")
            output[total] = distance
            total += 1
            print(total, " ", attack_name, " ",model_name)
            if (total >= max_check):
                np.save(model_name + "/" + attack_name + ".npy" ,output)
                break

        print("Time Taken = ", time.time() - start)
        # file.write("Time Taken = " + str(time.time() - start) + "\n")
        file.close()



def test_pgd(model_name):
    model = net().to(device)
    model_address = model_name + ".pt"
    model.load_state_dict(torch.load(model_address, map_location = device))
    print (model_name)
    attack = pgd_linf
    print ("pgd_linf")
    test_loader = DataLoader(mnist_test, batch_size = 1000, shuffle=False)

    start = time.time()
    epoch_i = 0
    lr = None
    # if (attack == pgd_all):
    #     adv_err, adv_loss = epoch_adversarial(test_loader, model_test, attack, device = device)
    # else:
    # adv_loss, adv_acc = epoch(test_loader, lr, model, epoch_i, device = device)
    # adv_loss, adv_acc = epoch_adversarial(test_loader, lr, model, epoch_i, attack, device = device, stop = True, num_iter = 100, restarts = 10)
    # print("Acc: ",adv_acc)
    # total_loss, total_acc_4 = epoch_adversarial(test_loader, None, model, epoch_i,  msd_v1, device = device, stop = True)
    total_loss, total_acc_1 = epoch_adversarial(test_loader,None,  model, epoch_i, pgd_l1_topk,device = device, stop = True, restarts = res)
    total_loss, total_acc_2 = epoch_adversarial(test_loader, None, model, epoch_i, pgd_l2, device = device, stop = True, restarts = res)
    total_loss, total_acc_inf = epoch_adversarial(test_loader, None, model, epoch_i, pgd_linf, device = device, stop = True, restarts = res)
    print('Test Acc 1: {0:.4f}'.format(total_acc_1))    
    print('Test Acc 2: {0:.4f}'.format(total_acc_2))    
    print('Test Acc Inf: {0:.4f}'.format(total_acc_inf))    
    # print('Test Acc Clean: {0:.4f}'.format(total_acc))    
    # print('Test Acc All: {0:.4f}'.format(total_acc_4))



# test_foolbox("Models/PGD_all_topk/pgd_all_const_eps_k_rand_alph_inf_0_01_iter_19", 1000,f)
# model_list = ["l_1_iter_14", "l_2_iter_14", "l_inf_iter_14", "msd_iter_14", "naive_all_out_iter_14", "naive_triple_iter_14"]
# model_list = ["msd_iter_14", "naive_all_out_iter_14", "naive_triple_iter_14"]
# choice = int(args[3])
# f = "logs/" + model_list[choice] + "_10restartsPA.txt"
# test_foolbox("Models/{0}".format(model_list[0]), 10, f)
# for i in range(11,20):
    # print (i)
# test_pgd("Models/msd_iter_14")
# test_pgd("Models/PGD_all_topk/pgd_all_const_eps_k_rand_alph_inf_0_01_k_limit_20_16Aug_iter_14")

# for model in model_list:
#     print (model)
#     test_pgd("Models/{0}".format(model))
    # break

# print ("Time Taken = ", time.time() - start)







def test_saver(model_name):
    eps_1 = [3,6,9,12,20,30,50,60,70,80,90,100]
    eps_2 = [0.1,0.2,0.3,0.5,1.0,1.5,2.0,2.5,3,5,7,10]
    eps_3 = [0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    num_1 = [50,50,50,100,100,200,200,200,300,300,300,300]
    num_2 = [30,40,50,50,100,100,150,150,150,150,300,300]
    num_3 = [30,40,50,50,100,100,150,150,150,150,300,300]
    attacks_l1 = torch.ones((batch_size, 12))*1000
    attacks_l2 = torch.ones((batch_size, 12))*1000
    attacks_linf = torch.ones((batch_size, 12))*1000
    model = net().to(device)
    model_address = model_name + ".pt"
    model.load_state_dict(torch.load(model_address, map_location = device))
    test_batches = DataLoader(mnist_test, batch_size = batch_size, shuffle=False)
    for index in range(len(eps_1)):
            e_1 = eps_1[index]
            n_1 = num_1[index]
            eps, total_acc_1 = epoch_adversarial_saver(batch_size, test_batches, model, pgd_l1_topk, e_1, n_1, device = device, restarts = res)
            attacks_l1[:,index] = eps
    attacks_l1 = torch.min(attacks_l1,dim = 1)[0]
    np.save(model_name + "/" + "CPGDL1" + ".npy" ,attacks_l1.numpy())

    for index in range(len(eps_2)):        
            e_2 = eps_2[index]
            n_2 = num_2[index]
            eps, total_acc_2 = epoch_adversarial_saver(batch_size, test_batches, model, pgd_l2, e_2, n_2, device = device, restarts = res)
            attacks_l2[:,index] = eps
    attacks_l2 = torch.min(attacks_l2,dim = 1)[0]
    np.save(model_name + "/" + "CPGDL2" + ".npy" ,attacks_l2.numpy())

    for index in range(len(eps_3)):
            e_3 = eps_3[index]
            n_3 = num_3[index]
            eps, total_acc_3 = epoch_adversarial_saver(batch_size, test_batches, model, pgd_linf, e_3, n_3, device = device, restarts = res)
            attacks_linf[:,index] = eps
    attacks_linf = torch.min(attacks_linf,dim = 1)[0]
    np.save(model_name + "/" + "CPGDLINF" + ".npy" ,attacks_linf.numpy())

model_list = ["LINF", "L1", "L2", "MSD_V0", "TRIPLE", "WORST", "VANILLA"]
model_name = "Selected/{}".format(model_list[choice])
# model_name = "Final/TRIPLE/lr1_iter_20_kmap1"
# model_name = "Final/MSD_V0/lr1_iter_20_kmap0_restartMSD"
# model_name = "Models/PGD_all_topk/pgd_all_const_eps_k_rand_alph_inf_0_01_k_limit_20_16Aug_iter_14"
# model_name = "Models/msd_iter_14"

if attack == 0:
    test_foolbox(model_name, 1000)
elif attack == 1:
    test_pgd(model_name)
else:
    test_saver(model_name)