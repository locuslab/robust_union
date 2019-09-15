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
folder = dataset + "/" + "Selected/" + model
if path != None:
	folder = path


out = open(dataset +"_RES/" + folder.split("/")[-1] + ".txt", "w")

def myprint(s):
	print(s)
	out.write(str(s) + "\n")


files = glob(folder + "/*.*")
# attacks_list = ['SAPA','PA','CPGDL1','IGD','AGNA','BA','DeepFool','PAL2','CPGDL2','FGSM','PGD','IGM', 'CPGDLINF']
attacks_list = ['SAPA','PA','CPGDL1','IGD','AGNA','AGNA','DeepFool','PAL2','CPGDL2','FGSM','PGD','IGM', 'CPGDLINF']
attacks_list = ['PA','PA','CPGDL1','IGD','AGNA','AGNA','DeepFool','DeepFool','CPGDL2','FGSM','PGD','PGD', 'CPGDLINF']
attacks_npy = []

l1_attacks = np.ones((num_samples, 3))
l2_attacks = np.ones((num_samples, 6))
linf_attacks = np.ones((num_samples, 4))

all_attacks = np.ones((num_samples, 3))

for a in attacks_list:
	y = np.load(folder + "/" + a+ ".npy")
	attacks_npy.append(y)


if dataset == "CIFAR10":
	l1 = [3,6,9,12,20,30,50,60,70,80,90,100]
	l2 =[0.05,0.1,0.2,0.3,0.5,0.7,1,2,3,4,5,10]
	linf = [0.005,0.01,0.02,0.03,0.05,0.07,0.1,0.2,0.3,0.4,0.5,1]
	e_l1 = 12
	e_l2 = 0.5
	e_linf = 0.03
else:
	l1 = [3,6,9,12,20,30,50,60,70,80,90,100]
	l2 = [0.1,0.2,0.3,0.5,1.0,1.5,2.0,2.5,3,5,7,10]
	linf = [0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
	e_l1 = 12
	e_l2 = 1.5
	e_linf = 0.3




def print_inference(eps, vals, name):
	steps = len(eps)
	accuracy = np.zeros((steps))
	
	myprint (name)
	# ipdb.set_trace()
	for i in range(steps):
		accuracy[i] = vals[vals>eps[i]].shape[0] / num_samples
		myprint(str(eps[i]) + " : " + str(accuracy[i]))
	out.write("\n")

def get_acc(pos, e):
	a = attacks_npy[pos]
	#ipdb.set_trace()
	name = attacks_list[pos]
	accuracy = a[a>e].shape[0]/num_samples
	myprint(name+ " : "+ str(accuracy))
	return a


l1_attacks[:,0] = get_acc(0,e_l1)
l1_attacks[:,1] = get_acc(1,e_l1)
l1_attacks[:,2] = get_acc(2,e_l1)

l1_attacks = np.amin(l1_attacks, axis = 1)

print_inference(l1, l1_attacks, "L1")



l2_attacks[:,0] = get_acc(3,e_l2)
l2_attacks[:,1] = get_acc(4,e_l2)
l2_attacks[:,2] = get_acc(5,e_l2)
l2_attacks[:,3] = get_acc(6,e_l2)
l2_attacks[:,4] = get_acc(7,e_l2)
l2_attacks[:,5] = get_acc(8,e_l2)

l2_attacks = np.amin(l2_attacks, axis = 1)
print_inference(l2, l2_attacks, "L2")

linf_attacks[:,0] = get_acc(9,e_linf)
linf_attacks[:,1] = get_acc(10,e_linf)
linf_attacks[:,2] = get_acc(11,e_linf)
linf_attacks[:,3] = get_acc(12,e_linf)

linf_attacks = np.amin(linf_attacks, axis = 1)

print_inference(linf, linf_attacks, "LINF")

# ipdb.set_trace()
all_attacks[:,0] = l1_attacks > e_l1
all_attacks[:,1] = l2_attacks > e_l2
all_attacks[:,2] = linf_attacks > e_linf

all_attacks = np.amin(all_attacks, axis = 1)
print ("All Attacks Acc = ", np.sum(all_attacks)/num_samples)
out.write("All Attacks Acc = " + str(np.sum(all_attacks)/num_samples))
out.close()
# print_inference(linf, vals = all_attacks)

# ipdb.set_trace()





