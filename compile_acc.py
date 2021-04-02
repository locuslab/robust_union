import numpy as np
import argparse
from glob import glob
import ipdb


parser = argparse.ArgumentParser(description='Draw inference from Results', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-model", help="Folder Containing All Attack Distances", type=str, default = "LINF")
parser.add_argument("-dataset", help = "MNIST or CIFAR 10", type = str, default = "MNIST")  # MNIST, CIFAR10
parser.add_argument("-num_samples", help = "Number of Test Examples", type = int, default = 1000)
parser.add_argument("-path", help = "Override model (custom path)", type = str, default = None)


params = parser.parse_args()

num_samples = params.num_samples
dataset = params.dataset
model = params.model
path = params.path
# folder = dataset + "/" + "Selected/" + model
folder = model
if path != None:
    folder = path


out = open(dataset +"_RES/" + folder.split("/")[-1] + ".txt", "w")

def myprint(s):
    print(s)
    out.write(str(s) + "\n")


files = glob(folder + "/*.*")
attacks_list_1 = ['CPGDL1','SAPA','PA']
attacks_list_2 = ['CPGDL2','IGD','AGNA','BA','DeepFool','PAL2','DDN', 'CWL2']
attacks_list_inf = ['CPGDLINF','FGSM','PGD','IGM']
attacks_npy_1 = []
attacks_npy_2 = []
attacks_npy_inf = []

l1_attacks = np.ones((num_samples, 3))
l2_attacks = np.ones((num_samples, 8))
linf_attacks = np.ones((num_samples, 4))

all_attacks = np.ones((num_samples, 3))
pall_attacks = np.ones((num_samples, 3))

for a in attacks_list_1:
    y = np.load(folder + "/" + a+ ".npy")
    attacks_npy_1.append(y)
for a in attacks_list_2:
    y = np.load(folder + "/" + a+ ".npy")
    attacks_npy_2.append(y)
for a in attacks_list_inf:
    y = np.load(folder + "/" + a+ ".npy")
    attacks_npy_inf.append(y)


if dataset == "CIFAR10":
    l1 = [3,6,(2000/255),12,20,30,50,60,70,80,90,100]
    # l1 = [3,6,8,10,12,20,30,50,70,80,90,100]
    l2 =[0.05,0.1,0.2,0.3,0.5,0.7,1,2,3,4,5,10]
    linf = [0.005,0.01,(4/255),0.03,0.05,0.07,0.1,0.2,0.3,0.4,0.5,1]
    # linf = [0.005,0.01,0.02,0.03,0.05,0.07,0.1,0.2,0.3,0.4,0.5,1]
    e_l1 = 12 #(2000/255)
    e_l2 = 0.5
    e_linf = 0.03 #(4/255)

    pe_l1 = (2000/255)
    pe_l2 = 0
    pe_linf = (4/255)
else:
    l1 = [3,6,(10),12,20,30,50,60,70,80,90,100]
    # l1 = [3,6,8,10,12,20,30,50,70,80,90,100]
    l2 = [0.1,0.2,0.3,0.5,1.0,1.5,2.0,2.5,3,5,7,10]
    linf = [0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    e_l1 = 12 #12
    e_l2 = 1.5 #1.5
    e_linf = 0.3

    pe_l1 = 10 #12
    pe_l2 = 2 #1.5
    pe_linf = 0.3


linf_dict = {'attacks_npy': attacks_npy_inf, 'attacks_list': attacks_list_inf, 'e': e_linf, 'pe': pe_linf}
l1_dict = {'attacks_npy': attacks_npy_1, 'attacks_list': attacks_list_1, 'e': e_l1, 'pe': pe_l1}
l2_dict = {'attacks_npy': attacks_npy_2, 'attacks_list': attacks_list_2, 'e': e_l2, 'pe': pe_l2}


def print_inference(eps, vals, name):
    steps = len(eps)
    accuracy = np.zeros((steps))
    
    myprint (name)
    # ipdb.set_trace()
    for i in range(steps):
        accuracy[i] = vals[vals>eps[i]].shape[0] / num_samples *100
        myprint(str(eps[i]) + " : " + str(accuracy[i]))
    out.write("\n")

def get_acc(pos, l_dict):
    a = l_dict['attacks_npy'][pos]
    e = l_dict['e']
    pe = l_dict['pe']
    #ipdb.set_trace()
    name = l_dict['attacks_list'][pos]
    accuracy = a[a>e].shape[0]/num_samples *100
    paccuracy = a[a>pe].shape[0]/num_samples *100
    myprint(name+ " : "+ str(accuracy)+ " | " + str(paccuracy) )

    return a



for i in range(len(attacks_list_inf)):
    linf_attacks[:,i] = get_acc(i,linf_dict)

linf_attacks = np.amin(linf_attacks, axis = 1)
print_inference(linf, linf_attacks, "LINF")


for i in range(len(attacks_list_2)):
    l2_attacks[:,i] = get_acc(i,l2_dict)

l2_attacks = np.amin(l2_attacks, axis = 1)
print_inference(l2, l2_attacks, "L2")

for i in range(len(attacks_list_1)):
    l1_attacks[:,i] = get_acc(i,l1_dict)

l1_attacks = np.amin(l1_attacks, axis = 1)
print_inference(l1, l1_attacks, "L1")


# ipdb.set_trace()
all_attacks[:,0] = l1_attacks > e_l1
all_attacks[:,1] = l2_attacks > e_l2
all_attacks[:,2] = linf_attacks > e_linf

all_attacks = np.amin(all_attacks, axis = 1)
print ("All Attacks Acc = ", np.sum(all_attacks)/num_samples)
out.write("All Attacks Acc = " + str(np.sum(all_attacks)/num_samples) + "\n")

pall_attacks[:,0] = l1_attacks > pe_l1
pall_attacks[:,1] = l2_attacks > pe_l2
pall_attacks[:,2] = linf_attacks > pe_linf

pall_attacks = np.amin(pall_attacks, axis = 1)
print ("Paper Compare All Attacks Acc = ", np.sum(pall_attacks)/num_samples)
out.write("All Attacks Acc = " + str(np.sum(pall_attacks)/num_samples) + "\n")
out.close()