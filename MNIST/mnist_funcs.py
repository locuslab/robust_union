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
import ipdb
import random


'''
#DEFAULTS
pgd_linf: epsilon=0.3, alpha=0.01, num_iter = 100
pgd_l0  : epsilon = 12, alpha = 1
pgd_l1_topk  : epsilon = 12, alpha = 0.02, num_iter = 100, k = rand(5,20) --> (alpha = alpha/k *20)
pgd_l2  : epsilon = 1.5, alpha=0.1, num_iter = 100


Original
def msd_v0(model, X,y, epsilon_l_inf = 0.3, epsilon_l_2= 1.5, epsilon_l_1 = 12, 
                        alpha_l_inf = 0.01, alpha_l_2 = 0.2, alpha_l_1 = 0.05, 
                        num_iter = 100, device = "cuda:1")
'''

def fgsm(model, X, y, epsilon=0.1):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

def norms(Z):
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]

def norms_l1(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:,None,None,None]

def norms_l0(Z):
    return ((Z.view(Z.shape[0], -1)!=0).sum(dim=1)[:,None,None,None]).float()

def norms_linf(Z):
    return Z.view(Z.shape[0], -1).abs().max(dim=1)[0]


def pgd_worst_dir(model, X,y, epsilon_l_inf = 0.3, epsilon_l_2= 1.5, epsilon_l_1 = 12, 
                                alpha_l_inf = 0.01, alpha_l_2 = 0.1, alpha_l_1 = 0.02, num_iter = 100, device = "cuda:1", k_map = 0):
    delta_1 = pgd_l1_topk(model, X, y, epsilon = epsilon_l_1, alpha = alpha_l_1, num_iter = 100, device = device, k_map = k_map)
    delta_2 = pgd_l2(model, X, y, epsilon = epsilon_l_2, alpha = alpha_l_2, num_iter = 100, device = device)
    delta_inf = pgd_linf(model, X, y, epsilon = epsilon_l_inf, alpha = alpha_l_inf, num_iter = 50, device = device, restarts = 2)
    
    batch_size = X.shape[0]

    loss_1 = nn.CrossEntropyLoss(reduction = 'none')(model(X + delta_1), y)
    loss_2 = nn.CrossEntropyLoss(reduction = 'none')(model(X + delta_2), y)
    loss_inf = nn.CrossEntropyLoss(reduction = 'none')(model(X + delta_inf), y)

    delta_1 = delta_1.view(batch_size,1,-1)
    delta_2 = delta_2.view(batch_size,1,-1)
    delta_inf = delta_inf.view(batch_size,1,-1)

    tensor_list = [loss_1, loss_2, loss_inf]
    delta_list = [delta_1, delta_2, delta_inf]
    loss_arr = torch.stack(tuple(tensor_list))
    delta_arr = torch.stack(tuple(delta_list))
    max_loss = loss_arr.max(dim = 0)
    
    # print(max_loss)

    delta = delta_arr[max_loss[1], torch.arange(batch_size), 0]
    delta = delta.view(batch_size,1, X.shape[2], X.shape[3])
    return delta


def msd_v0(model, X,y, epsilon_l_inf = 0.3, epsilon_l_2= 1.5, epsilon_l_1 = 12, 
                        alpha_l_inf = 0.01, alpha_l_2 = 0.07, alpha_l_1 = 0.03, 
                        num_iter = 50, device = "cuda:1", k_map = 0):
    alpha_l_1_default = alpha_l_1
    max_max_delta = torch.zeros_like(X)
    max_max_loss = torch.zeros(y.shape[0]).to(y.device)
    for j in range(1):
        # ipdb.set_trace()
        if j==0:
            delta = torch.zeros_like(X,requires_grad = True)
        else:
            delta = torch.rand_like(X,requires_grad = True) 
            delta.data = 2*delta.data - 1
        max_delta = torch.zeros_like(X)
        max_loss = torch.zeros(y.shape[0]).to(y.device)
        for t in range(num_iter):
            loss = nn.CrossEntropyLoss()(model(X + delta), y)
            loss.backward()
            with torch.no_grad():                
                #For L_2
                delta_l_2  = delta.data + alpha_l_2*delta.grad / norms(delta.grad)      
                delta_l_2  = torch.min(torch.max(delta_l_2, -X), 1-X) # clip X+delta to [0,1]
                delta_l_2 *= epsilon_l_2 / norms(delta_l_2).clamp(min=epsilon_l_2)

                #For L_inf
                delta_l_inf =  (delta.data + alpha_l_inf*delta.grad.sign()).clamp(-epsilon_l_inf,epsilon_l_inf)
                delta_l_inf = torch.min(torch.max(delta_l_inf, -X), 1-X) # clip X+delta to [0,1]

                #For L1
                if k_map == 0:
                    k = random.randint(5,20)
                    alpha_l_1   = (alpha_l_1_default/k)*20
                elif k_map == 1:
                    k = random.randint(10,40)
                    alpha_l_1   = (alpha_l_1_default/k)*40
                elif k_map ==2 :
                    k = 10
                    alpha_l_1 = 0.05
                else :
                    k = 1
                    alpha_l_1 = 0.1

                delta_l_1   = delta.data + alpha_l_1*l1_dir_topk(delta.grad, delta.data, X, alpha_l_1, k=k)
                delta_l_1   = proj_l1ball(delta_l_1, epsilon_l_1, device)
                delta_l_1   = torch.min(torch.max(delta_l_1, -X), 1-X) # clip X+delta to [0,1]
                
                #Compare
                delta_tup = (delta_l_inf, delta_l_1, delta_l_2)
                max_loss = torch.zeros(y.shape[0]).to(y.device)            
                selection_stats = torch.zeros(y.shape[0]).to(y.device)- 1
                for i,delta_temp in enumerate(delta_tup):
                    loss_temp = nn.CrossEntropyLoss(reduction = 'none')(model(X + delta_temp), y)
                    max_delta[loss_temp >= max_loss] = delta_temp[loss_temp >= max_loss]
                    selection_stats[loss_temp >= max_loss] = i
                    max_loss = torch.max(max_loss, loss_temp)
                delta.data = max_delta.data
                max_max_delta[max_loss> max_max_loss] = max_delta[max_loss> max_max_loss]
                max_max_loss[max_loss> max_max_loss] = max_loss[max_loss> max_max_loss]
            # ipdb.set_trace()
            # selection_stats_final = (selection_stats_final*(t-1) + selection_stats)/t
            delta.grad.zero_()
        del delta

    return max_max_delta

def msd_v1(model, X,y, epsilon_l_inf = 0.3, epsilon_l_2= 1.5, epsilon_l_1 = 12, 
                        alpha_l_inf = 0.01, alpha_l_2 = 0.1, alpha_l_1 = 0.02, 
                        num_iter = 100, device = "cuda:1", k_map = 0):
    alpha_l_1_default = alpha_l_1
    delta = torch.zeros_like(X,requires_grad = True)
    max_delta = torch.zeros_like(X)
    max_max_delta = torch.zeros_like(X)
    max_loss = torch.zeros(y.shape[0]).to(y.device)
    max_max_loss = torch.zeros(y.shape[0]).to(y.device)
    
    for t in range(num_iter):
        output = model(X+delta)
        incorrect = output.max(1)[1] != y 
        correct = (1-incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        with torch.no_grad():                
            #For L_2
            delta_l_2  = delta.data + alpha_l_2*correct*delta.grad / norms(delta.grad)      
            delta_l_2  = torch.min(torch.max(delta_l_2, -X), 1-X) # clip X+delta to [0,1]
            delta_l_2 *= epsilon_l_2 / norms(delta_l_2).clamp(min=epsilon_l_2)

            #For L_inf
            delta_l_inf =  (delta.data + alpha_l_inf*correct*delta.grad.sign()).clamp(-epsilon_l_inf,epsilon_l_inf)
            delta_l_inf = torch.min(torch.max(delta_l_inf, -X), 1-X) # clip X+delta to [0,1]

            #For L1
            if k_map == 0:
                k = random.randint(5,20)
                alpha_l_1   = (alpha_l_1_default/k)*20
            elif k_map == 1:
                k = random.randint(10,40)
                alpha_l_1   = (alpha_l_1_default/k)*40
            else:
                k = 20
                alpha_l_1 = alpha_l_1_default
            delta_l_1   = delta.data + alpha_l_1*correct*l1_dir_topk(delta.grad, delta.data, X, alpha_l_1, k=k)
            delta_l_1   = proj_l1ball(delta_l_1, epsilon_l_1, device)
            delta_l_1   = torch.min(torch.max(delta_l_1, -X), 1-X) # clip X+delta to [0,1]
            
            #Compare
            delta_tup = (delta_l_1, delta_l_2, delta_l_inf)
            max_loss = torch.zeros(y.shape[0]).to(y.device)            
            for delta_temp in delta_tup:
                loss_temp = nn.CrossEntropyLoss(reduction = 'none')(model(X + delta_temp), y)
                max_delta[loss_temp >= max_loss] = delta_temp[loss_temp >= max_loss]
                max_loss = torch.max(max_loss, loss_temp)
            delta.data = max_delta.data
            max_max_delta[max_loss> max_max_loss] = max_delta[max_loss> max_max_loss]
            max_max_loss[max_loss> max_max_loss] = max_loss[max_loss> max_max_loss]
        delta.grad.zero_()

    return max_max_delta

def pgd_linf(model, X, y, epsilon=0.3, alpha=0.01, num_iter = 100, restarts = 0, device = "cuda:1"):
    """ Construct FGSM adversarial examples on the examples X"""
    max_delta = torch.zeros_like(X)
    delta = torch.zeros_like(X, requires_grad=True)    
    for t in range(num_iter):
        output = model(X+delta)
        incorrect = output.max(1)[1] != y 
        correct = (1-incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
        #Finding the correct examples so as to attack only them
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta.data + alpha*correct*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
        delta.grad.zero_()
    max_delta = delta.detach()
    
    for i in range (restarts):
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = (delta.data * 2.0 - 1.0) * epsilon

        for t in range(num_iter):
            output = model(X+delta)
            incorrect = output.max(1)[1] != y 
            correct = (1-incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
            #Finding the correct examples so as to attack only them            
            loss = nn.CrossEntropyLoss()(model(X + delta), y)
            loss.backward()
            delta.data = (delta.data + alpha*correct*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
            delta.grad.zero_()

        output = model(X+delta)
        incorrect = output.max(1)[1] != y
        #Edit Max Delta only for successful attacks        
        max_delta[incorrect] = delta.detach()[incorrect]
    return max_delta



def pgd_l2(model, X, y, epsilon=2, alpha=0.1, num_iter = 100, restarts = 0, device = "cuda:1"):
    max_delta = torch.zeros_like(X)
    delta = torch.zeros_like(X, requires_grad = True)
    for t in range(num_iter):
        output = model(X+delta)
        incorrect = output.max(1)[1] != y 
        correct = (1-incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
        #Finding the correct examples so as to attack only them
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data +=  correct*alpha*delta.grad.detach() / norms(delta.grad.detach())
        delta.data *=  epsilon / norms(delta.detach()).clamp(min=epsilon)
        delta.data =   torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]     
        delta.grad.zero_()  

    max_delta = delta.detach()

    #restarts

    for k in range (restarts):
        delta = torch.rand_like(X, requires_grad=True) 
        delta.data *= (2.0*delta.data - 1.0)*epsilon 
        delta.data /= norms(delta.detach()).clamp(min=epsilon)
        for t in range(num_iter):
            output = model(X+delta)
            incorrect = output.max(1)[1] != y 
            correct = (1-incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
            #Finding the correct examples so as to attack only them
            loss = nn.CrossEntropyLoss()(model(X + delta), y)
            loss.backward()
            delta.data +=  correct*alpha*delta.grad.detach() / norms(delta.grad.detach())
            delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
            delta.grad.zero_()  

        output = model(X+delta)
        incorrect = output.max(1)[1] != y
        #Edit Max Delta only for successful attacks
        max_delta[incorrect] = delta.detach()[incorrect] 

    return max_delta    

def pgd_l0(model, X,y, epsilon = 12, alpha = 0.5, num_iter = 100, device = "cuda:1"):
    delta = torch.zeros_like(X, requires_grad = True)
    batch_size = X.shape[0]
    for t in range (epsilon):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        temp = delta.grad.view(batch_size, 1, -1)
        neg = (delta.data != 0)
        X_curr = X + delta
        neg1 = (delta.grad < 0)*(X_curr < 0.1)
        neg2 = (delta.grad > 0)*(X_curr > 0.9)
        neg += neg1 + neg2
        u = neg.view(batch_size,1,-1)
        temp[u] = 0
        my_delta = torch.zeros_like(X).view(batch_size, 1, -1)
        
        maxv =  temp.max(dim = 2)
        minv =  temp.min(dim = 2)
        val_max = maxv[0].view(batch_size)
        val_min = minv[0].view(batch_size)
        pos_max = maxv[1].view(batch_size)
        pos_min = minv[1].view(batch_size)
        select_max = (val_max.abs()>=val_min.abs()).float()
        select_min = (val_max.abs()<val_min.abs()).float()
        my_delta[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_max] = (1-X.view(batch_size, 1, -1)[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_max])*select_max
        my_delta[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_min] = -X.view(batch_size, 1, -1)[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_min]*select_min
        delta.data += my_delta.view(batch_size, 1, 28, 28)
        delta.grad.zero_()
    delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
    
    return delta.detach()


def pgd_l1_topk(model, X,y, epsilon = 12, alpha = 0.05, num_iter = 50, k_map = 0, device = "cuda:1", restarts = 1):
        #Gap : Dont attack pixels closer than the gap value to 0 or 1
    gap = alpha
    max_delta = torch.zeros_like(X)
    delta = torch.zeros_like(X, requires_grad = True)
    alpha_l_1_default = alpha

    for t in range (num_iter):
        output = model(X+delta)
        incorrect = output.max(1)[1] != y 
        correct = (1-incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
        #Finding the correct examples so as to attack only them
        loss = nn.CrossEntropyLoss()(model(X+delta), y)
        loss.backward()
        if k_map == 0:
            k = random.randint(5,20)
            alpha   = (alpha_l_1_default/k)*20
        elif k_map == 1:
            k = random.randint(10,40)
            alpha   = (alpha_l_1_default/k)*40
        else:
            k = 10
            alpha = alpha_l_1_default
        delta.data += alpha*correct*l1_dir_topk(delta.grad.detach(), delta.data, X, gap,k)
        if (norms_l1(delta) > epsilon).any():
            delta.data = proj_l1ball(delta.data, epsilon, device)
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1] 
        delta.grad.zero_() 

    max_delta = delta.detach()

    #Restarts    
    for k in range(restarts):
        delta = torch.rand_like(X,requires_grad = True)
        delta.data = (2*delta.data - 1.0)*epsilon 
        delta.data /= norms_l1(delta.detach()).clamp(min=epsilon)
        for t in range (num_iter):
            output = model(X+delta)
            incorrect = output.max(1)[1] != y 
            correct = (1-incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
            #Finding the correct examples so as to attack only them
            loss = nn.CrossEntropyLoss()(model(X+delta), y)
            loss.backward()
            if k_map == 0:
                k = random.randint(5,20)
                alpha   = (alpha_l_1_default/k)*20
            elif k_map == 1:
                k = random.randint(10,40)
                alpha   = (alpha_l_1_default/k)*40
            else:
                k = 20
                alpha = alpha_l_1_default
            delta.data += alpha*correct*l1_dir_topk(delta.grad.detach(), delta.data, X, gap,k)
            if (norms_l1(delta) > epsilon).any():
                delta.data = proj_l1ball(delta.data, epsilon, device)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1] 
            delta.grad.zero_() 
        output = model(X+delta)
        incorrect = output.max(1)[1] != y
        #Edit Max Delta only for successful attacks
        max_delta[incorrect] = delta.detach()[incorrect]   

    return max_delta

def kthlargest(tensor, k, dim=-1):
    val, idx = tensor.topk(k, dim = dim)
    return val[:,:,-1], idx[:,:,-1]

def l1_dir_topk(grad, delta, X, gap, k = 50) :
    #Check which all directions can still be increased such that
    #they haven't been clipped already and have scope of increasing
    # ipdb.set_trace()
    X_curr = X + delta
    batch_size = X.shape[0]
    channels = X.shape[1]
    pix = X.shape[2]
    # print (batch_size)
    neg1 = (grad < 0)*(X_curr <= gap)
#     neg1 = (grad < 0)*(X_curr == 0)
    neg2 = (grad > 0)*(X_curr >= 1-gap)
#     neg2 = (grad > 0)*(X_curr == 1)
    neg3 = X_curr <= 0
    neg4 = X_curr >= 1
    neg = neg1 + neg2 + neg3 + neg4
    u = neg.view(batch_size,1,-1)
    grad_check = grad.view(batch_size,1,-1)
    grad_check[u] = 0

    kval = kthlargest(grad_check.abs().float(), k, dim = 2)[0].unsqueeze(1)
    k_hot = (grad_check.abs() >= kval).float() * grad_check.sign()
    return k_hot.view(batch_size, channels, pix, pix)


def proj_l1ball(x, epsilon=10, device = "cuda:1"):
#     print (epsilon)
    # print (device)
    assert epsilon > 0
    # compute the vector of absolute values
    u = x.abs()
    if (u.sum(dim = (1,2,3)) <= epsilon).all():
        # print (u.sum(dim = (1,2,3)))
         # check if x is already a solution
        return x

    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    y = proj_simplex(u, s=epsilon, device = device)
    # compute the solution to the original problem on v
    y *= x.sign()
    y *= epsilon/norms_l1(y)
    return y


def proj_simplex(v, s=1, device = "cuda:1"):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    batch_size = v.shape[0]

    # check if we are already on the simplex    
    '''
    #Not checking this as we are calling this from the previous function only
    if v.sum(dim = (1,2,3)) == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    '''
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = v.view(batch_size,1,-1)
    n = u.shape[2]
    u, indices = torch.sort(u, descending = True)
    cssv = u.cumsum(dim = 2)
    # get the number of > 0 components of the optimal solution
    vec = u * torch.arange(1, n+1).float().to(device)
    comp = (vec > (cssv - s)).float()

    u = comp.cumsum(dim = 2)
    w = (comp-1).cumsum(dim = 2)
    u = u + w
    rho = torch.argmax(u, dim = 2)
    rho = rho.view(batch_size)
    c = torch.FloatTensor([cssv[i,0,rho[i]] for i in range( cssv.shape[0]) ]).to(device)
    c = c-s
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = torch.div(c,(rho.float() + 1))
    theta = theta.view(batch_size,1,1,1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clamp(min=0)
    return w


def epoch(loader, lr_schedule,  model, epoch_i = 0, criterion = nn.CrossEntropyLoss(), opt=None, device = "cuda:1", stop = False):
    """Standard training/evaluation epoch over the dataset"""
    train_loss = 0
    train_acc = 0
    train_n = 0

    for i, batch in enumerate(loader):
        X,y = batch[0].to(device), batch[1].to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            lr = lr_schedule(epoch_i + (i+1)/len(loader))
            opt.param_groups[0].update(lr=lr)
            opt.zero_grad()
            loss.backward()
            opt.step()

        train_loss += loss.item()*y.size(0)
        train_acc += (yp.max(1)[1] == y).sum().item()
        train_n += y.size(0)

        if stop:
            break
        
    return train_loss / train_n, train_acc / train_n

def epoch_adversarial(loader, lr_schedule, model, epoch_i, attack, criterion = nn.CrossEntropyLoss(), 
                        opt=None, device = "cuda:1", stop = False, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    train_loss = 0
    train_acc = 0
    train_n = 0

    for i, batch in enumerate(loader):
        X,y = batch[0].to(device), batch[1].to(device)
        # if attack == pgd_all_old:
        #     delta = attack(model, X, y, device = device, **kwargs)
        #     delta = delta[0]
        # else:
        delta = attack(model, X, y, device = device, **kwargs)
        yp = model(X+delta)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            lr = lr_schedule(epoch_i + (i+1)/len(loader))
            opt.param_groups[0].update(lr=lr)
            opt.zero_grad()
            loss.backward()
            opt.step()

        train_loss += loss.item()*y.size(0)
        train_acc += (yp.max(1)[1] == y).sum().item()
        train_n += y.size(0)
        if stop:
            break
        
    return train_loss / train_n, train_acc / train_n

def epoch_adversarial_saver(batch_size, loader, model, attack, epsilon, num_iter, device = "cuda:0", restarts = 10):
    # ipdb.set_trace()
    criterion = nn.CrossEntropyLoss()
    train_loss = 0
    train_acc = 0
    train_n = 0
    # print("Attack: ", attack, " epsilon: ", epsilon )

    for i,batch in enumerate(loader): 
        X,y = batch[0].to(device), batch[1].to(device)
        delta = attack(model, X, y, epsilon = epsilon, num_iter = num_iter, device = device, restarts = restarts)
        output = model(X+delta)
        loss = criterion(output, y)
        train_loss += loss.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        correct = (output.max(1)[1] == y).float()
        eps = (correct*1000 + epsilon - 0.000001).float()
        train_n += y.size(0)
        break
    return eps,  train_acc / train_n

def epoch_adversarial_tracker(loader, lr_schedule, model, epoch_i, attack, criterion = nn.CrossEntropyLoss(), 
                        opt=None, device = "cuda:1", stop = False, **kwargs):

    """Adversarial training/evaluation epoch over the dataset"""
    train_loss = 0
    train_acc = 0
    train_n = 0
    percentage= [0,0,0]
    
    for i, batch in enumerate(loader):
        X,y = batch[0].to(device), batch[1].to(device)
        # if attack == pgd_all_old:
        #     delta, iter_l1, iter_l2, iter_linf = attack(model, X, y, device = device, **kwargs)
        #     percentage[0]+=iter_l1
        #     percentage[1]+=iter_l2
        #     percentage[2]+=iter_linf
        # else:
        delta = attack(model, X, y, device = device, **kwargs)
        yp = model(X+delta)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            lr = lr_schedule(epoch_i + (i+1)/len(loader))
            opt.param_groups[0].update(lr=lr)
            opt.zero_grad()
            loss.backward()
            opt.step()
        

        train_loss += loss.item()*y.size(0)
        train_acc += (yp.max(1)[1] == y).sum().item()
        train_n += y.size(0)

    num_iter = sum(percentage)
    # if (attack == pgd_all_old):
    #     print ("L_1 = ", (percentage[0]/num_iter), " L_2 = ", percentage[1]/num_iter, " L_inf =", percentage[2]/num_iter )

    return train_loss / train_n, train_acc / train_n


def triple_adv(loader, lr_schedule, model, epoch_i, attack, criterion = nn.CrossEntropyLoss(), 
                    opt=None, device= "cuda:1", epsilon_l_1 = 12, epsilon_l_2 = 1.5, epsilon_l_inf = 0.3, num_iter = 100, k_map = 0):
    train_loss = 0
    train_acc = 0
    train_n = 0
    for i, batch in enumerate(loader):
        X,y = batch[0].to(device), batch[1].to(device)
        if opt:
            lr = lr_schedule(epoch_i + (i+1)/len(loader))
            opt.param_groups[0].update(lr=lr)

        X,y = X.to(device), y.to(device)
        #L1
        delta = pgd_l1_topk(model, X, y, device = device, epsilon = epsilon_l_1, k_map = k_map)
        yp = model(X+delta)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        train_loss += loss.item()*y.size(0)
        train_acc += (yp.max(1)[1] == y).sum().item()
        train_n += y.size(0)
        


        #L2
        delta = pgd_l2(model, X, y, device = device, epsilon = epsilon_l_2)
        yp = model(X+delta)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        train_loss += loss.item()*y.size(0)
        train_acc += (yp.max(1)[1] == y).sum().item()
        train_n += y.size(0)

        
        #Linf
        delta = pgd_linf(model, X, y, device = device, epsilon = epsilon_l_inf)
        yp = model(X+delta)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        train_loss += loss.item()*y.size(0)
        train_acc += (yp.max(1)[1] == y).sum().item()
        train_n += y.size(0)
        # break
    return train_loss / train_n, train_acc / train_n

def correctness(y,yp):
    return (y[y==yp.max(dim=1)[1]].shape[0])

def plot_images(X,y,yp,M,N):
    f,ax = plt.subplots(M,N, sharex=True, sharey=True, figsize=(N,M*1.3))
    correct = 0
    for i in range(M):
        for j in range(N):
            ax[i][j].imshow(1-X[i*N+j][0].cpu().numpy(), cmap="gray")
            title = ax[i][j].set_title("Pred: {}".format(yp[i*N+j].max(dim=0)[1]))
            plt.setp(title, color=('g' if yp[i*N+j].max(dim=0)[1] == y[i*N+j] else 'r'))
            ax[i][j].set_axis_off()
            if (yp[i*N+j].max(dim=0)[1] == y[i*N+j] ):
                        correct +=1
    plt.tight_layout()
    print ("Correct = ", correct)


'''
#########OBSOLETE########


def pgd_all_old(model, X,y, randomness = 0.04, epsilon_l_inf = 0.3, epsilon_l_2= 1.5, epsilon_l_1 = 12, alpha_l_inf = 0.01, alpha_l_2 = 0.1, alpha_l_1 = 0.02, num_iter = 100, device = "cuda:1"):
    percentage = [0,0,0]
    delta = torch.zeros_like(X,requires_grad = True)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        # print (norms_linf(delta))
        # print (norms_l1(delta))
        # print (norms(delta))
        temp_grad = delta.grad.detach()
        temp_data = delta.data
        #For L_inf
        delta_l_inf=  (temp_data + alpha_l_inf*temp_grad.sign()).clamp(-epsilon_l_inf,epsilon_l_inf)
        delta_l_inf = torch.min(torch.max(delta_l_inf, -X), 1-X) # clip X+delta to [0,1]
        #For L_2
        delta_l_2  = temp_data + alpha_l_2*temp_grad / norms(temp_grad)
        delta_l_2 *= epsilon_l_2 / norms(delta_l_2).clamp(min=epsilon_l_2)
        delta_l_2  = torch.min(torch.max(delta_l_2, -X), 1-X) # clip X+delta to [0,1]
        #For L1
        delta_l_1  = temp_data + alpha_l_1*l1_dir_topk(temp_grad, delta.data, X, alpha_l_1, k = 50)
        delta_l_1 = proj_l1ball(delta_l_1, epsilon_l_1, device)
        # delta_l_1 *= epsilon_l_1/norms_l1(delta_l_1) 
        delta_l_1  = torch.min(torch.max(delta_l_1, -X), 1-X) # clip X+delta to [0,1]
        #Compare
        delta_tup = (delta_l_1, delta_l_2, delta_l_inf)
        attack_str_l = ["delta_l_1", "delta_l_2", "delta_inf"]
        losses_list = [0,0,0]
        delta_temp_list = []
        max_loss = -1*float("inf")
        n = 0
        delta_curr = delta
        for delta_temp in delta_tup:
            delta_curr.data = delta_temp
            delta_temp_list.append(delta_temp)
            loss_temp = nn.CrossEntropyLoss()(model(X + delta_curr), y)
            losses_list[n] = loss_temp
            # print ("Loss: ", loss_temp.item(), " Attack: ", attack_str_l[n], )
            ##############IF NOT RANDOM ##################
            if (loss_temp > max_loss):
                max_loss = loss_temp
                delta.data = delta_temp
                correct = n
            ##############################################
            n += 1
        ################IF RANDOM######################
        # max_loss = max(losses_list)
        # min_loss = min(losses_list)
        # if (max_loss - min_loss) < randomness:
        #     correct = np.random.randint(3)
        # else:
        #     correct = np.argmax(losses_list)
        # delta.data = delta_temp_list[correct]
        ################IF RANDOM######################

        percentage[correct] += 1
        delta.grad.zero_()
    # print ("L_1 = ", (percentage[0]/num_iter), " L_2 = ", percentage[1]/num_iter, " L_inf =", percentage[2]/num_iter )

    return delta.detach(), percentage[0], percentage[1], percentage[2]


    def pgd_l1(model, X,y, epsilon = 12, alpha = 0.1, num_iter = 100, device = "cuda:1"):

    delta = torch.zeros_like(X, requires_grad = True)
    # print ("Here")
    for t in range (num_iter):
        loss = nn.CrossEntropyLoss()(model(X+delta), y)
        loss.backward()
        # print ("Yes")
        delta.data += alpha*l1_dir(delta.grad.detach(), delta.data, X, alpha)
        delta.data = proj_l1ball(delta.data, epsilon, device)
        delta.data *= epsilon/norms_l1(delta.data)
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1] 
        delta.grad.zero_()    
    return delta.detach()
''
def l1_dir(grad, delta, X, alpha) :
    #Check which all directions can still be increased such that 
    #they haven't been clipped already and have scope of increasing
    X_curr = X + delta
    batch_size = X.shape[0]
    # print (batch_size)
    neg1 = (grad < 0)*(X_curr < alpha)
#     neg1 = (grad < 0)*(X_curr == 0)
    neg2 = (grad > 0)*(X_curr >1-alpha)
#     neg2 = (grad > 0)*(X_curr == 1)
    neg3 = X_curr < 0
    neg4 = X_curr > 1
    neg = neg1 + neg2 + neg3 + neg4
    u = neg.view(batch_size,1,-1)
    grad_check = grad.view(batch_size,1,-1)
    grad_check[u] = 0

    max_dir =  grad_check.max(dim = 2)
    min_dir =  grad_check.min(dim = 2)
    val_max = max_dir[0].view(batch_size)
    val_min = min_dir[0].view(batch_size)
    pos_max = max_dir[1].view(batch_size)
    pos_min = min_dir[1].view(batch_size)
    select_max = (val_max.abs()>=val_min.abs()).float()
    select_min = (val_max.abs()<val_min.abs()).float()

    one_hot = torch.zeros_like(grad_check)
    one_hot[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_max] = 1*select_max
    one_hot[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_min] = -1*select_min
    one_hot = one_hot.view(batch_size,1,28,28)
    
    #Return this direction
    return one_hot
  

'''