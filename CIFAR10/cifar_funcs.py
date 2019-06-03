import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch
import ipdb

def pgd_linf(model, X, y, epsilon=0.03, alpha=0.003, num_iter = 40, randomize=False, device = "cuda:0"):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)  
#     ipdb.set_trace()
    for t in range(num_iter):
#         print (type(X), X.shape, type(delta), delta.shape)
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta.data + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,255]
        delta.grad.zero_()
    return delta.detach()


def norms_l0(Z):
    return ((Z.view(Z.shape[0], -1)!=0).sum(dim=1)[:,None,None,None]).float()


def pgd_l0(model, X,y, epsilon = 12, alpha = 1, num_iter = 0, device = "cuda:1"):
    delta = torch.zeros_like(X, requires_grad = True)
    batch_size = X.shape[0]
    # print("Updated")
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
        select_max = (val_max.abs()>=val_min.abs()).half()
        select_min = (val_max.abs()<val_min.abs()).half()
        my_delta[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_max] = (1-X.view(batch_size, 1, -1)[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_max])*select_max
        my_delta[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_min] = -X.view(batch_size, 1, -1)[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_min]*select_min
        delta.data += my_delta.view(batch_size, 3, 32, 32)
        delta.grad.zero_()
    delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
    
    return delta.detach()

def norms_l1(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:,None,None,None]

def pgd_l1(model, X,y, epsilon = 12, alpha = 0.1, num_iter = 150, device = "cuda:1"):
    gap = alpha
    delta = torch.zeros_like(X, requires_grad = True)
    # print ("Here")
    for t in range (num_iter):
        loss = nn.CrossEntropyLoss()(model(X+delta), y)
        loss.backward()
        # print ("Yes")
        delta.data += alpha*l1_dir(delta.grad.detach(), delta.data, X, gap)
#         delta.data *= epsilon/norms_l1(delta.data)
        delta.data = proj_l1ball(delta.data, epsilon, device)
#         if t==0:
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1] 
        delta.grad.zero_()    
#     delta.data *= epsilon/norms_l1(delta.data)
#     delta.data = torch.min(torch.max(delta.detach(), -X), 1-X)
    return delta.detach()



def l1_dir(grad, delta, X, gap) :
    #Check which all directions can still be increased sunum_iterch that 
    #they haven't been clipped already and have scope of increasing
    X_curr = X + delta
    batch_size = X.shape[0]
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

    max_dir =  grad_check.max(dim = 2)
    min_dir =  grad_check.min(dim = 2)
    val_max = max_dir[0].view(batch_size)
    val_min = min_dir[0].view(batch_size)
    pos_max = max_dir[1].view(batch_size)
    pos_min = min_dir[1].view(batch_size)
    select_max = (val_max.abs()>=val_min.abs()).half()
    select_min = (val_max.abs()<val_min.abs()).half()

    one_hot = torch.zeros_like(grad_check)
    one_hot[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_max] = 1*select_max
    one_hot[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_min] = -1*select_min
    one_hot = one_hot.view(batch_size,3,32,32)
    
    #Return this direction
    return one_hot
  

def proj_l1ball(x, epsilon=10, device = "cuda:1"):
#     print (epsilon)
    # print (device)
    assert epsilon > 0
#     ipdb.set_trace()
    # compute the vector of absolute values
    u = x.abs()
    if (u.sum(dim = (1,2,3)) <= epsilon).all():
        # print (u.sum(dim = (1,2,3)))
         # check if x is already a solution
#         y = x* epsilon/norms_l1(x)
        return x

    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    y = proj_simplex(u, s=epsilon, device = device)
    # compute the solution to the original problem on v
    y = y.view(-1,3,32,32)
    y *= x.sign()
#     y *= epsilon/norms_l1(y)
    return y


def proj_simplex(v, s=1, device = "cuda:1"):
#     ipdb.set_trace()
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
    vec = u * torch.arange(1, n+1).half().to(device)
    comp = (vec > (cssv - s)).half()
    ###################NOT THE MOST EFFICIENT WAY###########3
    u = comp.cumsum(dim = 2)
    w = (comp-1).cumsum(dim = 2)
    u = u + w
    rho = torch.argmax(u, dim = 2)
    rho = rho.view(batch_size)
    c = torch.HalfTensor([cssv[i,0,rho[i]] for i in range( cssv.shape[0]) ]).to(device)
    c = c-s
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = torch.div(c,(rho.half() + 1))
    theta = theta.view(batch_size,1,1,1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clamp(min=0)
    return w


# In[11]:


# torch.save(model.state_dict(), "vanila_cifar10.pt")

def norms(Z):
#     ipdb.set_trace()
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]

def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     img = img*255
    npimg = img.cpu().numpy()
    npimg = np.transpose(npimg, (1,2,0)).astype('float32')
    print ("TYPE: ", type(npimg), " SHAPE: ", npimg.shape, " DTYPE: ", npimg.dtype)
    plt.imshow(npimg)
    plt.show()
    
def pgd_l2(model, X, y, epsilon=0.3, alpha=0.05, num_iter = 40, device = "cuda:0"):
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data +=  alpha*delta.grad.detach() / norms(delta.grad.detach())
        delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
        
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
        
        delta.grad.zero_()        
    return delta.detach()



def epoch(loader, lr_schedule,  model, epoch_i, criterion = nn.CrossEntropyLoss(), opt=None, device = "cuda:0"):
    """Standard training/evaluation epoch over the dataset"""
    train_loss = 0
    train_acc = 0
    train_n = 0
    for i,batch in enumerate(loader): 
        X,y = batch['input'], batch['target']
#         print (X.shape)
#         print (type(X))
#         X = transform_train(X)
        output = model(X)
        loss = criterion(output, y)        
        if opt != None:   
            lr = lr_schedule(epoch_i + (i+1)/len(loader))
            opt.param_groups[0].update(lr=lr)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        train_loss += loss.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)
        
    return train_loss / train_n, train_acc / train_n


def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     img = img*255
    npimg = img.cpu().numpy()
    npimg = np.transpose(npimg, (1,2,0)).astype('float32')
    print ("TYPE: ", type(npimg), " SHAPE: ", npimg.shape, " DTYPE: ", npimg.dtype)
    plt.imshow(npimg)
    plt.show()

def epoch_adversarial_saver(loader, model, attack, epsilon, num_iter, device = "cuda:0"):
    criterion = nn.CrossEntropyLoss()
    train_loss = 0
    train_acc = 0
    train_n = 0
    print("Attack: ", attack, " epsilon: ", epsilon )
    for i,batch in enumerate(loader): 
        X,y = batch['input'], batch['target']
        delta = attack(model, X, y, epsilon = epsilon, num_iter = num_iter, device = device)
        output = model(X+delta)
        loss = criterion(output, y)
        train_loss += loss.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        print(output.max(1)[1] == y)
        train_n += y.size(0)
        break
    return train_loss / train_n, train_acc / train_n

def epoch_adversarial(loader, lr_schedule, model, epoch_i, attack, criterion = nn.CrossEntropyLoss(), 
    opt=None, device = "cuda:0", stop = False, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    train_loss = 0
    train_acc = 0
    train_n = 0
#     ipdb.set_trace()
    
    for i,batch in enumerate(loader): 
        X,y = batch['input'], batch['target']
        delta = attack(model, X, y, device = device, **kwargs)
        output = model(X+delta)
        # imshow(X[11])
        # print (X[11])
        # imshow((X+delta)[11])
        # print (norms_l1(delta))
#         output = model(X)
        loss = criterion(output, y)
        train_loss += loss.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)
        if opt != None:   
            lr = lr_schedule(epoch_i + (i+1)/len(loader))
            opt.param_groups[0].update(lr=lr)
            opt.zero_grad()
            loss.backward()
            opt.step()
        else:
            if (stop):
                break
        
#         break
        
    return train_loss / train_n, train_acc / train_n

def epoch_triple_adv(loader, lr_schedule, model, epoch_i, attack,  criterion = nn.CrossEntropyLoss(),
                     opt=None, device= "cuda:0", epsilon_l_1 = 12, epsilon_l_2 = 0.3, epsilon_l_inf = 0.03, num_iter = 50):
    
    train_loss = 0
    train_acc = 0
    train_n = 0

    for i,batch in enumerate(loader): 
        X,y = batch['input'], batch['target']
        lr = lr_schedule(epoch_i + (i+1)/len(loader))
        opt.param_groups[0].update(lr=lr)

        #L1
        delta = pgd_l1(model, X, y, device = device, epsilon = epsilon_l_1)
        output = model(X+delta)
        loss = criterion(output,y)
        train_loss += loss.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)
        
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        #L2
        delta = pgd_l2(model, X, y, device = device, epsilon = epsilon_l_2)
        output = model(X+delta)
        loss = nn.CrossEntropyLoss()(output,y)
        train_loss += loss.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        

        #Linf
        delta = pgd_linf(model, X, y, device = device, epsilon = epsilon_l_inf)
        output = model(X+delta)
        loss = nn.CrossEntropyLoss()(output,y)
        train_loss += loss.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        else:
            break
        # break
    return train_loss / train_n, train_acc / train_n




def pgd_all(model, X,y, epsilon_l_inf = 0.03, epsilon_l_2= 0.3, epsilon_l_1 = 12, 
                alpha_l_inf = 0.003, alpha_l_2 = 0.05, alpha_l_1 = 0.1, num_iter = 50, device = "cuda:0"):
    percentage = [0,0,0]
    delta = torch.zeros_like(X,requires_grad = True)
    max_delta = torch.zeros_like(X)
    max_max_delta = torch.zeros_like(X)
    max_loss = torch.zeros(y.shape[0]).to(y.device).half()
    max_max_loss = torch.zeros(y.shape[0]).to(y.device).half()
    
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        with torch.no_grad():                
            #For L_2
            delta_l_2  = delta.data + alpha_l_2*delta.grad / norms(delta.grad)      
            # delta_l_2 *= epsilon_l_2 / norms(delta_l_2)
            delta_l_2 *= epsilon_l_2 / norms(delta_l_2).clamp(min=epsilon_l_2)
            delta_l_2  = torch.min(torch.max(delta_l_2, -X), 1-X) # clip X+delta to [0,1]

            #For L_inf
            delta_l_inf=  (delta.data + alpha_l_inf*delta.grad.sign()).clamp(-epsilon_l_inf,epsilon_l_inf)
            delta_l_inf = torch.min(torch.max(delta_l_inf, -X), 1-X) # clip X+delta to [0,1]

            #For L1
            delta_l_1  = delta.data + alpha_l_1*l1_dir(delta.grad, delta.data, X, alpha_l_1)
            delta_l_1 = proj_l1ball(delta_l_1, epsilon_l_1, device)
            # delta_l_1 *= epsilon_l_1/norms_l1(delta_l_1) 
            delta_l_1  = torch.min(torch.max(delta_l_1, -X), 1-X) # clip X+delta to [0,1]
            
            #Compare
            delta_tup = (delta_l_1, delta_l_2, delta_l_inf)
            max_loss = torch.zeros(y.shape[0]).to(y.device).half()
            # ipdb.set_trace()        
            for delta_temp in delta_tup:
                loss_temp = nn.CrossEntropyLoss(reduction = 'none')(model(X + delta_temp), y)
                max_delta[loss_temp >= max_loss] = delta_temp[loss_temp >= max_loss]
                max_loss = torch.max(max_loss, loss_temp)
            delta.data = max_delta.data
            max_max_delta[max_loss> max_max_loss] = max_delta[max_loss> max_max_loss]
            max_max_loss[max_loss> max_max_loss] = max_loss[max_loss> max_max_loss]
        delta.grad.zero_()

    return max_max_delta


def pgd_all_out(model, X,y, epsilon_l_inf = 0.03, epsilon_l_2= 0.3, epsilon_l_1 = 12, 
    alpha_l_inf = 0.003, alpha_l_2 = 0.05, alpha_l_1 = 0.1, num_iter = 50, device = "cuda:0"):
    delta_1 = pgd_l1(model, X, y, epsilon = epsilon_l_1, alpha = alpha_l_1,  device = device)
    delta_2 = pgd_l2(model, X, y, epsilon = epsilon_l_2, alpha = alpha_l_2,  device = device)
    delta_inf = pgd_linf(model, X, y, epsilon = epsilon_l_inf, alpha = alpha_l_inf, device = device)
    
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
    delta = delta.view(batch_size,3, X.shape[2], X.shape[3])
    return delta

def pgd_all_old(model, X,y, epsilon_l_inf = 0.03, epsilon_l_2= 0.3, epsilon_l_1 = 12, alpha_l_inf = 0.003, alpha_l_2 = 0.1, alpha_l_1 = 0.3, num_iter = 50, device = "cuda:1"):
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
        delta_l_1  = temp_data + alpha_l_1*l1_dir(temp_grad, delta.data, X, alpha_l_1)
        delta_l_1 = proj_l1ball(delta_l_1, epsilon_l_1, device)
        delta_l_1 *= epsilon_l_1/norms_l1(delta_l_1) 
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


        percentage[correct] += 1
        delta.grad.zero_()
    # print ("L_1 = ", (percentage[0]/num_iter), " L_2 = ", percentage[1]/num_iter, " L_inf =", percentage[2]/num_iter )

    return delta.detach()

