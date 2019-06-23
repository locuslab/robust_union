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
from my_funcs import *
import time

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)    


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



def test_model(model_name, max_tests,f):
    file = open(f,"a")
    print (model_name)
    torch.manual_seed(0)
    model_test = nn.Sequential(nn.Conv2d(1, 32, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2),
                          nn.Conv2d(32, 64, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2),
                          Flatten(),
                          nn.Linear(7*7*64, 1024), nn.ReLU(),
                          nn.Linear(1024, 10)).to(device)
    model_address = model_name + ".pt"
    model_test.load_state_dict(torch.load(model_address, map_location = device))
    model_test.eval()
    fmodel = foolbox.models.PyTorchModel(model_test,   # return logits in shape (bs, n_classes)
                                         bounds=(0., 1.), num_classes=10,
                                         device=device)
    # ipdb.set_trace()
    # b, l = get_batch(bs=batch_size)
    # if not model_test.has_grad: 
    #     print ("No Grad!!")
    #     GE = foolbox.gradient_estimators.CoordinateWiseGradientEstimator(0.1)
    #     fmodel = foolbox.models.ModelWithEstimatedGradients(fmodel, GE)

    attacks_list = ['PGD']
    # attacks_list = ['SAPA','PA','IGD','AGNA','BA','DeepFool','PAL2','FGSM','IFGSM','PGD','IGM']
    types_list   = [ 3  ]#  ,  2      , 2    , 3]
    # types_list   = [ 0    , 0  , 2   , 2    , 2  ,  2      , 2    , 3    , 3     , 3   , 3   ]
    # attacks_list = ['SAPA']#,'PA','IGD','AGNA','BA','DeepFool','PAL2','FGSM','IFGSM','PGD','IGM']
    for i in range(len(attacks_list)):
        attack_name = attacks_list[i]
        types = types_list[i]
        max_check = max_tests
        test_loader = DataLoader(mnist_test, batch_size = 1, shuffle=False)

        if attack_name == "BA":
            max_check = min(100,max_tests)
            test_loader = DataLoader(mnist_test, batch_size = 1, shuffle=True)
        start = time.time()
        file.write ("\n" + attack_name + "\n")
        attack, metric, args, kwargs = get_attack(attack_name, fmodel)
        total = 0
        err = 0
        for X,y in test_loader:
            total += 1
            image  = X[0,0,:,:].view(1,28,28).detach().numpy()
            label  = y[0].item()
            adversarial = attack(image, label=label, **kwargs)
            try :
                adversarial.all()
            except:
                file.write(str(1000) + "\n")
                continue
            adv = torch.from_numpy(adversarial).view(1,1,28,28).to(device)
            pred_label = torch.argmax(model_test(adv),dim = 1)[0]
            if (label != pred_label):
                err+=1
            adv = torch.from_numpy(adversarial)
            # linf_d = torch.abs(X - adv).max().item()
            # l2_d  = norms(X - adv).item()
            # l1_d = norms_l1(X - adv).item()
            # l0_d = norms_l0(X - adv).item()
            # print(total," Label = ", label, " Pred = ", pred_label, " lo_d = ", l0_d, " linf_d = ", linf_d, " l2_d = ", l2_d, " l1_d = ", l1_d)
            if (types == 0):
                file.write(str(norms_l0(X - adv).item()) + "\n")
            elif (types == 2):
                file.write(str(norms(X - adv).item()) + "\n")
            elif (types == 3):
                file.write(str(torch.abs(X - adv).max().item()) + "\n")
            if (total >= max_check):
                break
        print("Time Taken = ", time.time() - start)
    file.close()

start = time.time()
mnist_test = datasets.MNIST("../../data", train=False, download=True, transform=transforms.ToTensor())
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# test_model("NN/April/8th/rand_0.04_alphas_[0.2, 0.25, 0.05]_iter_7", 100)
import sys
args = sys.argv
f = args[1]
test_model("NN/9March/1_1_1_random_2_extend_new_iter_6", 10000,f)
# test_model("NN/April/8th/rand_0.04_alphas_[0.2, 0.25, 0.05]_iter_7", 1000,f)
# test_model("NN/Naive/all_out_iter_5", 10000,f)
print ("Time Taken = ", time.time() - start)
