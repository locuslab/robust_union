## Pretrained Models  
Pretrained models for each of the training methods discussed in the paper are available at [this link](https://drive.google.com/open?id=1aCNQdoSnUs7mpT7zIRu0n56DThLlv98V)
Please download these models in the folder named `Selected`  in order to test the models.
The testing code is automatically designed to pick the models from that folder. 


## Testing Code

+ `test.py` - Test the Adversarially Robust Models
  > `gpu_id`  - Id of GPU to be used  - `default = 0`  
  > `model`   - Type of Adversarial Training:  - `default = 3`   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0: l_inf  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1: l_1  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2: l_2   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3: msd  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4: avg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5: max  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6: vanilla  
  > `batch_size` - Batch Size for Test Set -`default = 100`  
  > `attack` - Foolbox = 0; Custom PGD = 1, Min PGD = 2, Fast DDN = 3;  - `default = 0`  
  > `restarts`  - Number of Random Restarts - `default = 10`  
  > `path` - To override default model fetching - `default = None`   
  > `subset` - Subset for Foolbox attacks - `default = -1`   


## Training Code

+ `train.py` - Train the Adversarially Robust Models
  > `gpu_id`  - Id of GPU to be used  - `default = 0`  
  > `model`   - Type of Adversarial Training:  - `default = 3`   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0: l_inf  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1: l_1  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2: l_2  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3: msd  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4: avg   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5: max   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6: vanilla  
  > `batch_size` - Batch Size for Train Set -`default = 100`  
  > `alpha_l_1` - Step Size for L1 attacks - `default = 1.0`    
  > `alpha_l_2` - Step Size for L2 attacks - `default = 0.1`   
  > `alpha_l_inf`- Step Size for Linf attacks - `default = 0.01`   
  > `epsilon_l_1` - Perturb Radius for L1 attacks - `default = 10`    
  > `epsilon_l_2` - Perturb Radius for L2 attacks - `default = 2.0`   
  > `epsilon_l_inf`- Perturb Radius for Linf attacks - `default = 0.3`    
  > `num_iter`  - PGD iterations - `default = 100`   
  > `epochs`  - Number of Epochs - `default = 15`  
  > `seed`  - Random Seed - `default = 0`  


## For Reproducing the Results

**0. P_inf:**  
`python train.py -model 0 -epochs 20`
  
**1. P1:**  
`python train.py -model 1 -epochs 20`

**2. P2:**  
`python train.py -model 2 -epochs 20`
  
**3. MSD:**   
`python train.py -model 3 -alpha_l_1 0.8 -seed 107`

**4. AVG:**  
`python train.py -model 4 -alpha_l_2 0.2 -epsilon_l_1 12.0`

**5. MAX:**  
`python train.py -model 5`
  Restarts for l1 = 2 (Manually set)
  Early stop at epoch 4
  


