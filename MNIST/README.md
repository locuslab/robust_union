## Pretrained Models  
Pretrained models for each of the training methods discussed in the paper are available in the folder `Selected`  
The testing code is automatically designed to pick the models from the folder. 


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
  > `attack` - Foolbox = 0; Custom PGD = 1, Min PGD = 2, Fast DDN = 3;  - `default = 1`  
  > `restarts` 	- Number of Random Restarts - `default = 10`  
  > `path` - To override default model fetching - `default = 12`   
  > `subset` - Subset for Foolbox attacks - `default = -1`   


## Training Code

+ `train.py` - Train the Adversarially Robust Models
  > `gpu_id`  - Id of GPU to be used  - `default = 0`  
  > `model`   - Type of Adversarial Training:  - `default = 3`   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0: l_inf  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1: l_1  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2: l_2   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3: msd  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4: triple  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5: worst  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6: vanilla  
  > `batch_size` - Batch Size for Train Set -`default = 100`  
  > `alpha_l_1` - Step Size for L1 attacks - `default = 0.05`    
  > `alpha_l_2` - Step Size for L2 attacks - `default = 0.1`   
  > `alpha_l_inf`- Step Size for Linf attacks - `default = 0.01`    
  > `num_iter`  - PGD iterations - `default = 100`   
  > `epochs`  - Number of Epochs - `default = 15`  


## For Reproducing the Results

**0. P_inf:**  
`python train.py -model 0`
	
**1. P1:**  
`python train.py -model 1`

**2. P2:**  
`python train.py -model 2`
	
**3. MSD:**   
`python train.py -model 3 -epochs 15 -alpha_l_2 0.2`

**4. Worst Augmentation:**  
`python train.py -model 4 -epochs 20 -alpha_l_1 0.02`  
	Restarts for l1 = 2
	
**5. Triple Augmentation:**  
`python train.py -model 5 -epochs 20`

