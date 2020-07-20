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


def trainer(params, device_id, batch_size, choice, alpha_l_1, alpha_l_2, alpha_l_inf, num_iter, epochs, epsilon_l_1, epsilon_l_2, epsilon_l_inf, lr_mode, msd_initialization, smallest_adv, n, opt_type, lr_max, resume, resume_iter,seed, randomize, k_map):

    mnist_train = datasets.MNIST("../../data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("../../data", train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(mnist_train, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size = 1000, shuffle=False)


    device = torch.device("cuda:{0}".format(device_id) if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.shape[0], -1)    

    def net():
        return nn.Sequential(nn.Conv2d(1, 32, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(32, 64, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2), Flatten(), nn.Linear(7*7*64, 1024), nn.ReLU(), nn.Linear(1024, 10))

    def net_tanh():
        return nn.Sequential(nn.Conv2d(1, 32, 5, padding = 2), nn.Tanh(), nn.MaxPool2d(2, 2), nn.Conv2d(32, 64, 5, padding = 2), nn.Tanh(), nn.MaxPool2d(2, 2), Flatten(), nn.Linear(7*7*64, 1024), nn.Tanh(), nn.Linear(1024, 10))

    attack_list = [ pgd_linf ,  pgd_l1_topk,   pgd_l2 ,  msd_v0 ,  triple_adv ,  pgd_worst_dir, msd_v0]#TRIPLE, VANILLA DON'T HAVE A ATTACK NAME ANYTHING WORKS
    attack_name = ["pgd_linf", "pgd_l1_topk", "pgd_l2", "msd_v0", "triple_adv", "pgd_worst_dir", "vanilla"]
    folder_name = ["LINF", "L1", "L2", "MSD", "AVG", "MAX", "VANILLA"]


    def myprint(a):
        print(a)
        file.write(a)
        file.write("\n")
        file.flush()

    attack = attack_list[choice]
    name = attack_name[choice]
    folder = folder_name[choice]

    print (name)
    criterion = nn.CrossEntropyLoss()

    #### TRAIN CODE #####
    root = f"Models/{folder_name[choice]}"
    import glob, os, json
    # if(not os.path.exists(root)):
    #     num = 0
    # else:
    #     files = glob.glob(f"{root}/model_*")
    #     files = [int(x.split("_")[-1]) for x in files]
    #     num = max(files) + 1 if len(files) > 0 else 0
    num = n
    model_dir = f"{root}/model_{num}"

    if(not os.path.exists(model_dir)):
        os.makedirs(model_dir)
    file = open(f"{model_dir}/logs.txt", "a")    
    with open(f"{model_dir}/model_info.txt", "w") as f:
        json.dump(params.__dict__, f, indent=2)

    lr_schedule = lambda t: np.interp([t], [0, epochs*2//5, epochs], [0, lr_max, 0])[0]
    # k_map = 0
    
    if lr_mode != None:
    	if lr_mode == 1:
    		lr_schedule = lambda t: np.interp([t], [0, 3, 10, epochs], [0, 0.05, 0.001, 0.0001])[0]
    		# k_map = 0
    	elif lr_mode == 2:
    		# k_map = 0
    		lr_schedule = lambda t: np.interp([t], [0, epochs*2//5, epochs*4//5, epochs], [0, lr_max, lr_max/10, 0])[0]

    if activation == "tanh":
        model = net_tanh().to(device)
    else:
        model = net().to(device)

    if opt_type == "SGD":
        opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    else:
        opt = optim.Adam(model.parameters(), lr=0.1)
    t_start = 1

    if resume:
        location = f"{model_dir}/iter_{str(resume_iter)}.pt"
        t_start = resume_iter + 1
        model.load_state_dict(torch.load(location, map_location = device))

    for t in range(t_start,epochs+1):
        start = time.time()
        print ("Learning Rate = ", lr_schedule(t))
        if choice == 6:
            train_loss, train_acc = epoch(train_loader, lr_schedule, model, epoch_i = t, opt = opt, device = device)
        elif choice == 4:
            train_loss, train_acc = triple_adv(train_loader, lr_schedule, model, epoch_i = t, attack = attack, 
            												opt = opt, device = device, k_map = k_map, 
            												alpha_l_inf = alpha_l_inf, alpha_l_2 = alpha_l_2, alpha_l_1 = alpha_l_1, 
            												num_iter = num_iter, epsilon_l_1 = epsilon_l_1, epsilon_l_2 = epsilon_l_2, 
            												epsilon_l_inf = epsilon_l_inf, randomize = randomize)
        elif choice in [3]:
            train_loss, train_acc = epoch_adversarial(train_loader, lr_schedule, model, epoch_i = t, attack = attack, 
                                                            opt = opt, device = device, k_map = k_map, 
                                                            alpha_l_inf = alpha_l_inf, alpha_l_2 = alpha_l_2, alpha_l_1 = alpha_l_1, 
                                                            num_iter = num_iter, epsilon_l_1 = epsilon_l_1, epsilon_l_2 = epsilon_l_2, 
                                                            epsilon_l_inf = epsilon_l_inf, msd_init = msd_initialization, randomize = randomize)
        elif choice in [5]:
            train_loss, train_acc = epoch_adversarial(train_loader, lr_schedule, model, epoch_i = t, attack = attack, 
                                                            opt = opt, device = device, k_map = k_map, 
                                                            alpha_l_inf = alpha_l_inf, alpha_l_2 = alpha_l_2, alpha_l_1 = alpha_l_1, 
                                                            num_iter = num_iter, epsilon_l_1 = epsilon_l_1, epsilon_l_2 = epsilon_l_2, epsilon_l_inf = epsilon_l_inf, randomize = randomize)
        
        elif choice == 1:
            train_loss, train_acc = epoch_adversarial(train_loader, lr_schedule, model, epoch_i = t, attack = attack, opt = opt, device = device, k_map = k_map, randomize = randomize)
        else:
            train_loss, train_acc = epoch_adversarial(train_loader, lr_schedule, model, epoch_i = t, attack = attack, opt = opt, device = device, randomize = randomize)

        test_loss, test_acc = epoch(test_loader, lr_schedule,  model, epoch_i = t,  device = device, stop = True)
        linf_loss, linf_acc = epoch_adversarial(test_loader, lr_schedule,  model, epoch_i = t, attack = pgd_linf, device = device, stop = True, epsilon = epsilon_l_inf)
        l2_loss, l2_acc = epoch_adversarial(test_loader, lr_schedule,  model, epoch_i = t, attack = pgd_l2, device = device, stop = True, epsilon = epsilon_l_2)
        l1_loss, l1_acc_topk = epoch_adversarial(test_loader, lr_schedule,  model, epoch_i = t, attack = pgd_l1_topk, device = device, stop = True, epsilon = epsilon_l_1)
        time_elapsed = time.time()-start
        myprint(f'Epoch: {t}, Loss: {train_loss:.4f} Train : {train_acc:.4f} Clean : {test_acc:.4f}, Test  1: {l1_acc_topk:.4f}, Test  2: {l2_acc:.4f}, Test  inf: {linf_acc:.4f}, Time: {time_elapsed:.1f}, Model: {model_dir}')    
        
        torch.save(model.state_dict(), "{0}/iter_{1}.pt".format(model_dir, str(t)))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adversarial Training for MNIST', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-gpu_id", help="Id of GPU to be used", type=int, default = 0)
    parser.add_argument("-model", help="Type of Adversarial Training: \n\t 0: l_inf \n\t 1: l_1 \n\t 2: l_2 \n\t 3: msd \n\t 4: avg \n\t 5: max \n\t 6: vanilla", type=int, default = 3)
    parser.add_argument("-batch_size", help = "Batch Size for Train Set (Default = 100)", type = int, default = 100)
    parser.add_argument("-alpha_l_1", help = "Step Size for L1 attacks", type = float, default = 1.0)
    parser.add_argument("-alpha_l_2", help = "Step Size for L2 attacks", type = float, default = 0.1)
    parser.add_argument("-alpha_l_inf", help = "Step Size for Linf attacks", type = float, default = 0.01)
    parser.add_argument("-num_iter", help = "PGD iterations", type = int, default = 100)
    parser.add_argument("-epochs", help = "Number of Epochs", type = int, default = 15)
    parser.add_argument("-epsilon_l_1", help = "Step Size for L1 attacks", type = float, default = 10)
    parser.add_argument("-epsilon_l_2", help = "Epsilon Radius for L2 attacks", type = float, default = 2)
    parser.add_argument("-epsilon_l_inf", help = "Epsilon Radius for Linf attacks", type = float, default = 0.3)
    parser.add_argument("-lr_mode", help = "Specify LR Mode Manually", type = int, default = None)
    parser.add_argument("-msd_initialization", help = "0 for 0 start, 1 for random init, 2 for both", type = int, default = 1)
    parser.add_argument("-smallest_adv", help = "Early stop at closest adversarial example", type = int, default = 1)
    parser.add_argument("-model_id", help = "For Saving", type = int, default = 0)
    parser.add_argument("-opt_type", help = "For Saving", type = str, default = "Adam")
    parser.add_argument("-activation", help = "relu/tanh", type = str, default = "relu")
    parser.add_argument("-lr_max", help = "For Saving", type = float, default = 1e-3)
    parser.add_argument("-resume", help = "For Saving", type = int, default = 0)
    parser.add_argument("-resume_iter", help = "For Saving", type = int, default = -1)
    parser.add_argument("-seed", help = "Seed", type = int, default = 0)
    parser.add_argument("-randomize", help = "Seed", type = int, default = 0)
    parser.add_argument("-k_map", help = "K for L1 attack", type = int, default = 0)
    params = parser.parse_args()
    device_id = params.gpu_id
    batch_size = params.batch_size
    choice = params.model
    alpha_l_1 = params.alpha_l_1
    alpha_l_2 = params.alpha_l_2
    alpha_l_inf = params.alpha_l_inf
    num_iter = params.num_iter
    epochs = params.epochs
    epsilon_l_1 = params.epsilon_l_1
    epsilon_l_2 = params.epsilon_l_2
    epsilon_l_inf = params.epsilon_l_inf
    lr_mode = params.lr_mode
    msd_initialization = params.msd_initialization
    smallest_adv = params.smallest_adv
    n = params.model_id
    opt_type = params.opt_type
    lr_max = params.lr_max
    resume = params.resume
    resume_iter = params.resume_iter
    activation = params.activation
    seed = params.seed
    randomize = params.randomize
    k_map = params.k_map
    trainer(params, device_id, batch_size, choice, alpha_l_1, alpha_l_2, alpha_l_inf, num_iter, epochs, epsilon_l_1, epsilon_l_2, epsilon_l_inf, lr_mode, msd_initialization, smallest_adv, n, opt_type, lr_max, resume, resume_iter, seed, randomize, k_map)
