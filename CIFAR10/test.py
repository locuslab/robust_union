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
import time
sys.path.append('./utils/')
from core import *
from torch_backend import *
import ipdb
import sys 
import foolbox
import foolbox.attacks as fa
from cifar_funcs import *

args = sys.argv

device_id = args[1]
model_name = args[2]
device = torch.device("cuda:{0}".format(device_id) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(int(device_id))

torch.cuda.device_count() 


epochs = 50
DATA_DIR = './data'
dataset = cifar10(DATA_DIR)
t = Timer()

# cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
# cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

print('Preprocessing test data')
test_set = list(zip(transpose(normalise2(dataset['test']['data'])), dataset['test']['labels']))
print('Finished in {0:.2} seconds'.format(t()))




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

eps_1 = [3,6,9,12,20,30,50,60,70,80,90,100]
eps_2 = [0.05,0.1,0.2,0.3,0.5,0.7,1,2,3,4,5,10]
eps_3 = [0.005,0.01,0.02,0.03,0.05,0.07,0.1,0.2,0.3,0.4,0.5,1]
num_1 = [150,150,150,300,500,500,500,600,700,800,900,1000]
num_2 = [30,40,50,50,100,100,150,150,150,150,300,300]
num_3 = [30,40,50,50,100,100,150,150,150,150,300,300]

def test_pgd(model_name):
    print (model_name)
    batch_size = 100
    test_batches = Batches(test_set, batch_size, shuffle=False, num_workers=2, gpu_id = torch.cuda.current_device())
    print(device)
    model = PreActResNet18().to(device)
    for m in model.children(): 
        if not isinstance(m, nn.BatchNorm2d):
            m.half()   
            
    criterion = nn.CrossEntropyLoss()

    import time
    start_time = time.time()

    attack = pgd_all
    model.load_state_dict(torch.load(model_name, map_location = device))
    model.eval()
    lr_schedule = lambda t: np.interp([t], [0, epochs*2//5, epochs*4//5, epochs], [0, 0.1, 0.005, 0])[0]
    epoch_i = 0
    try:
        total_loss, total_acc = epoch(test_batches, lr_schedule, model, epoch_i, criterion, opt = None, device = device, stop = False)
    except:
        total_loss, total_acc = epoch(test_batches, lr_schedule, model, epoch_i, criterion, opt = None, device = device, stop = False)
        print("ok")
    total_loss, total_acc_4 = epoch_adversarial(test_batches, None, model, epoch_i,  pgd_all, device = device, stop = False)
    total_loss, total_acc_1 = epoch_adversarial(test_batches,None,  model, epoch_i, pgd_l1_topk,device = device, stop = False, restarts = 10)
    total_loss, total_acc_2 = epoch_adversarial(test_batches, None, model, epoch_i, pgd_l2, device = device, stop = False, restarts = 10)
    total_loss, total_acc_inf = epoch_adversarial(test_batches, None, model, epoch_i, pgd_linf, device = device, stop = False, restarts = 10)
    print('Test Acc 1: {0:.4f}'.format(total_acc_1))    
    print('Test Acc 2: {0:.4f}'.format(total_acc_2))    
    print('Test Acc Inf: {0:.4f}'.format(total_acc_inf))    
    print('Test Acc Clean: {0:.4f}'.format(total_acc))    
    print('Test Acc All: {0:.4f}'.format(total_acc_4))    

    '''    
    for index in range(len(eps_1)):
        e_1 = eps_1[index]
        # e_2 = eps_2[index]
        # e_3 = eps_3[index]
        n_1 = num_1[index]
        # n_2 = num_2[index]
        # n_3 = num_3[index]
        total_loss, total_acc_1 = epoch_adversarial_saver(test_batches, model, pgd_l1, e_1, n_1, device = device)
        print('Test Acc 1: {0:.4f}'.format(total_acc_1))    
        # total_loss, total_acc_2 = epoch_adversarial_saver(test_batches, model, pgd_l2, e_2, n_2, device = device)
        # total_loss, total_acc_3 = epoch_adversarial_saver(test_batches, model, pgd_linf, e_3, n_3, device = device)
        # break
        # total_loss, total_acc_0 = epoch_adversarial(test_batches, lr_schedule, model, epoch_i,  pgd_l0, criterion, opt = None, device = device, stop = False)
    '''
    # print('Test Acc Clean: {0:.4f}, Test Acc 1: {1:.4f}, Test Acc 2: {2:.4f}, Test Acc inf: {3:.4f}, Time: {4:.1f}'.format(test_acc, total_acc_1,total_acc_2, total_acc_3, time.time() - start_time))    
    # print('Test Acc 0: {0:.4f}'.format(total_acc_0))    



def test_foolbox(model_name, max_tests, f):

    file = open(f,'a')
    model = PreActResNet18().cuda()
    for m in model.children(): 
        if not isinstance(m, nn.BatchNorm2d):
            m.half()   
            
    model.load_state_dict(torch.load(model_name, map_location = device))
    criterion = nn.CrossEntropyLoss()

    model.eval()    
    for m in model.children(): 
        if not isinstance(m, nn.BatchNorm2d):
            m.float() 
    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10, device = device)

    lr_schedule = lambda t: np.interp([t], [0, epochs*2//5, epochs*4//5, epochs], [0, 0.1, 0.005, 0])[0]
    epoch_i = 0

    # types_list   = [ 2   , 2    ,   2      , 2    , 3    , 3     , 3   , 3 , 0, 0  ]

    # types_list   = [ 0, 0]
    # types_list   = [ 2   , 2    , 2  ,  2      , 2   ]
    # types_list   = [ 3    , 3     , 3   , 3  ]
    types_list   = [ 2]

    # attacks_list = ['SAPA', 'PA']
    # attacks_list = ['IGD','AGNA','BA','DeepFool','PAL2']
    # attacks_list = ['FGSM','IFGSM','PGD','IGM'] 
    attacks_list = ['BA'] 
    
    for j in range(len(attacks_list)):
        test_batches = Batches(test_set, batch_size = 1, shuffle=False, num_workers=2, gpu_id = torch.cuda.current_device())

        attack_name = attacks_list[j]
        print (attack_name)
        types = types_list[j]
        max_check = max_tests

        if attack_name == "BA":
            max_check = min(100,max_tests)
            test_batches = Batches(test_set, batch_size = 1, shuffle=False, num_workers=2, gpu_id = torch.cuda.current_device())

        start = time.time()
        file.write ("\n" + attack_name + "\n")
        attack, metric, args, kwargs = get_attack(attack_name, fmodel)
        total = 0
        err = 0
        corr = 0
        for i,batch in enumerate(test_batches): 
            X,y = batch['input'].float(), batch['target'].float()
            total+=1
            print(total)
            image = X[0,:,:,:].view(3,32,32).detach().cpu().numpy().astype('float32')
            # ipdb.set_trace()
            label = y[0].long().item()
            distance = 1000
            restarts = 1
            for r in range (restarts):
                try:
                    adversarial = attack(image, label=label)# , iterations = 100, epsilon = 0.3, stepsize =0.1)
                except:
                    if (i == 0):
                        adversarial = attack(image, label=label)# , iterations = 100, epsilon = 0.3, stepsize =0.1)
                    else:
                        print ("assertion error")
                        # file.write("1000\n")
                        continue
                try :
                    adversarial.all()
                    adv = torch.from_numpy(adversarial).float().view(1,3,32,32).to(device)
                    distance = min(distance, norms_l2(X - adv).item())
                except:
                    # file.write("1000\n")
                    a = 0
                    # print(1000)
                    continue

            file.write(str(distance) + "\n")
            
            # pred_label = torch.argmax(model(adv),dim = 1)[0]
            # if (label != pred_label):
            #     err+=1
            #     if (types == 0):
            #         file.write(str(norms_l0(X - adv).item()) + "\n")
            #     elif (types == 2):
            #         file.write(str(norms(X - adv).item()) + "\n")
            #         # print(str(norms(X - adv).item()))
            #     elif (types == 3):
            #         file.write(str(torch.abs(X - adv).max().item()) + "\n")
            # else:
            #     file.write("1000\n")
                # print(1000)
            if (total >= max_check):
                break
        print("Time Taken = ", time.time() - start)
    file.close()


model_list = ["worst_topk/lr1_iter_50.pt", "triple_topk/lr1_iter_50.pt", "L1_topk/lr1_iter_50.pt", "msd_topk/lr1_topk_rand_iter_50.pt"]

# choice = int(args[3])
# f = "logs/" + model_list[choice].split("/")[-1] + "_10restartsPA.txt"
# test_foolbox("RobustModels/{0}".format(model_list[choice]), 100,f)



eps_1 = [3,6,9,12,20,30,50,60,70,80,90,100]
eps_2 = [0.05,0.1,0.2,0.3,0.5,0.7,1,2,3,4,5,10]
eps_3 = [0.005,0.01,0.02,0.03,0.05,0.07,0.1,0.2,0.3,0.4,0.5,1]
num_1 = [50,50,50,100,100,200,200,200,300,300,300,300]
num_2 = [30,40,50,50,100,100,150,150,150,150,300,300]
num_3 = [30,40,50,50,100,100,150,150,150,150,300,300]


def test_saver(model_name):
    model = PreActResNet18().cuda()
    for m in model.children(): 
        if not isinstance(m, nn.BatchNorm2d):
            m.half()   
            
    model.load_state_dict(torch.load(model_name, map_location = device))
    criterion = nn.CrossEntropyLoss()

    model.eval()    

    batch_size = 1000
    test_batches = Batches(test_set, batch_size, shuffle=False, num_workers=2, gpu_id = torch.cuda.current_device())

    try:
        total_loss, total_acc = epoch(test_batches, None, model, 0, criterion, opt = None, device = device, stop = True)
    except:
        print ("OK")

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
  



models = ["l_2_0_3.pt", "l_inf_0_03.pt", "worst_topk/lr1_iter_50.pt", "triple_topk/lr1_iter_50.pt", "L1_topk/lr1_iter_50.pt", "lr1_topk_rand_iter_50.pt"]

# for model_name in models:
    # print(model_name)
# test_pgd("RobustModels/L1_corrected/l_1_att2_iter_25.pt")
f = args[3]
# test_pgd("RobustModels/worst_topk/lr1_iter_50.pt")
# test_pgd("RobustModels/triple_topk/lr1_iter_50.pt")
# test_pgd("RobustModels/L1_topk/lr1_iter_50.pt")
# test_pgd("RobustModels/{0}".format(models[0]))
# test_pgd("RobustModels/{0}".format(models[1]))
# test_pgd("RobustModels/L1_topk/lr1_iter_50.pt")
test_pgd("RobustModels/msd_topk/lr1_topk_rand_iter_50.pt")
# start = time.time()
import time
start = time.time()
# test_foolbox("RobustModels/{0}".format(model_list[3]), 10, f)
# print("Time taken = ",time.time() - start)
# test_foolbox("RobustModels/triple_topk/lr1_iter_50.pt", 1000, f)
# test_foolbox("RobustModels/L1_topk/lr1_iter_50.pt", 1000, f)
# test_foolbox("RobustModels/msd_topk/lr1_topk_rand_iter_50.pt", 1000, f)

# test_pgd("RobustModels/worst_topk/lr2_iter_50.pt")
# test_foolbox("Madry/cifar10_challenge/models/model_0/checkpoint", 100, f)
# test_pgd("RobustModels/msd_topk/lr1_topk_rand_iter_45.pt")
'''
folder_name = ["L1_topk", "msd_topk", "worst_topk", "triple_topk"]
for choice in range(4):
    # choice = int(args[2])
    print(folder_name[choice])
    test_pgd("RobustModels/{0}/lr1_iter_50.pt".format(folder_name[choice]))
'''

    # break
# attack = args[2]
# for i in range(5,51,5):
#     model_name = attack + "_iter_" + str(i) + ".pt"
#     # model_name = model_name[4:]
#     test_pgd(model_name)
# test_foolbox("RobustModels/L1/l_1_att2_iter_25.pt", 1000, args[3])

