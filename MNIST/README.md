## Training Code

+ `train.py` - Train the Adversarially Robust Models
  > `gpu_id` 	- Id of GPU to be used  - `default = 0`  
  > `model` 	- Type of Adversarial Training:  - `default = 3`   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0: l_inf  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1: l_1  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2: l_2   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3: msd  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4: triple  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5: worst  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6: vanilla  
  > `batch_size` - Batch Size for Train Set -`default = 100`  
  > `lr_schedule` - Scheduler Choice (see code) - `default = 1`  
  > `k_map` 	- Choice for L1 attacks - `default = 0`  
  > `epsilon_l_1` - Epsilon for L1 attacks - `default = 12`   
  > `epsilon_l_2` - Epsilon for L2 attacks - `default = 1.5`   
  > `epsilon_l_inf` - Epsilon for Linf attacks - `default = 0.3`    
  > `alpha_l_1`	- Step Size for L1 attacks - `default = 0.05`    
  > `alpha_l_2`	- Step Size for L2 attacks - `default = 0.2`   
  > `alpha_l_inf`- Step Size for Linf attacks - `default = 0.01`    
  > `num_iter` 	- PGD iterations - `default = 100`   
  > `epochs` 	- Number of Epochs - `default = 15`  


## Testing Code

+ `test.py` - Test the Adversarially Robust Models
  > `gpu_id` 	- Id of GPU to be used  - `default = 0`  
  > `model` 	- Type of Adversarial Training:  - `default = 3`   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0: l_inf  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1: l_1  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2: l_2   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3: msd  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4: triple  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5: worst  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6: vanilla  
  > `batch_size` - Batch Size for Test Set -`default = 100`  
  > `attack` - Foolbox = 0; Custom PGD = 1; Saving PGD = 2;  - `default = 1`  
  > `restarts` 	- Number of Random Restarts - `default = 10`  
  > `path` - To override default model fetching - `default = 12`   
  > `subset` - Subset for Foolbox attacks - `default = -1`   


## For Reproducing the Results

	1. MSD MNIST: 
	opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
	lr_schedule = lambda t: np.interp([t], [0, 3, 7, 15], [0, 0.05, 0.001, 0.0001])[0]
	criterion = nn.CrossEntropyLoss()
	epochs = 15
	epoch_adversarial(train_loader, lr_schedule, model, epoch_i = t, attack = pgd_all, opt = opt, device = device, epsilon_l_inf = 0.3, epsilon_l_1 = 12, epsilon_l_2 = 1.5)
	In pgd_all, epsilon_l_inf = 0.3, 
            	epsilon_l_2= 1.5, 
            	epsilon_l_1 = 12, 
                alpha_l_inf = 0.01,
                k = random.randint(5,20)
            	alpha_l_1 = 0.05/k*20
            	alpha_l_2 = 0.2, 
		        num_iter = 100

	2. MNIST Triple Augmentation:

	3. MNIST Worst Augmentation:

	4. MNIST P1:

	5. MNIST P2:

	6. MNIST P_inf:


