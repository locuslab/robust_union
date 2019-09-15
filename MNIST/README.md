## Robustness Against Multiple Perturbations on the MNIST Dataset


### Training the model
`python3 train.py '


### Testing the code
Run the folowing commands
`python3 test.py'

## Code for MNIST Adversarial Union Robustness

### Reproducing the results:
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


