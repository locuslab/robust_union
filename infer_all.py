import numpy as np
import argparse
from glob import glob

parser = argparse.ArgumentParser(description='Draw inference from Results', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("folder", help="Folder Containing All Attack Distances", type=str)
parser.add_argument("num_samples", help = "Number of Test Examples", type = int)
parser.add_argument("lp_ball", help = "Type of Attack Ball", type = "str", choices=['l1', 'l2', 'linf'])
parser.add_argument("dataset", help = "MNIST or CIFAR 10", type = "str", choices=['MNIST', 'CIFAR'])


files = glob.glob(folder + "/*.*")
attacks_list = ['SAPA','PA','IGD','AGNA','BA','DeepFool','PAL2','FGSM','IFGSM','PGD','IGM']

params = parser.parse_args()

if datset == "CIFAR":
	l1 = [3,6,9,12,20,30,50,60,70,80,90,100]
	l2 =[0.05,0.1,0.2,0.3,0.5,0.7,1,2,3,4,5,10]
	linf = [0.005,0.01,0.02,0.03,0.05,0.07,0.1,0.2,0.3,0.4,0.5,1]
else:
	l1 = [3,6,9,12,20,30,50,60,70,80,90,100]
	l2 = [0.1,0.2,0.3,0.5,1.0,1.5,2.0,2.5,3,5,7,10]
	linf = [0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

vals = np.load(params.file_id)

steps = len(l1)
accuracy = np.zeros((steps))

if lp_ball == 'l1':
	eps = l1
elif lp_ball == 'l2':
	eps = l2
else:
	eps = linf

out = open("inference" + params.file_id.split["."][0] + ".txt")

for i in range(steps):
	accuracy[i] = vals[vals>eps[i]].shape[0]
	out.write(accuracy[i])
	out.write("\n")


np.save("infer" + file_id, accuracy)


