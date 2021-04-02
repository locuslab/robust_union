# Adversarial Robustness Against the Union of Multiple Perturbation Models

Repository for the paper [Adversarial Robustness Against the Union of Multiple Perturbation Models](https://arxiv.org/abs/1909.04068) by [Pratyush Maini](https://pratyushmaini.github.io), [Eric Wong](https://riceric22.github.io) and [Zico Kolter](http://zicokolter.com)

## What is robustness against a union of multiple perturbation models?
While there has been a significant body of work that has focussed on creating classifiers that are robust to adversarial perturbations within a specified ℓp ball, the majority of defences have been restricted to a particular norm. In this work we focus on developing models that are robust against multiple ℓp balls simultaneously, namely ℓ∞, ℓ2, and ℓ1 balls.


## What does this work provide?
We show that it is indeed possible to adversarially train a robust model against a union of norm-bounded attacks, by using a natural generalization of the standard PGD-based procedure for adversarial training to multiple threat models. With this approach, we are able to train standard architectures which are robust against ℓ∞, ℓ2, and ℓ1 attacks, outperforming past approaches on the MNIST dataset and providing the first CIFAR10 network trained to be simultaneously robust against (ℓ∞,ℓ2,ℓ1) threat models, which achieves adversarial accuracy rates of (47.6%,64.3%,53.4%) for (ℓ∞,ℓ2,ℓ1) perturbations with radius ϵ=(0.03,0.5,12).

<p align="center">
  <img align="center" src="https://pratyushmaini.github.io/files/MSD.gif" width="250" height="250" />
</p>

## What does this repository contain?
Code for training and evaluating all the experiments that support the aforementioned paper are provided in this repository. 
The instructions for reproducing the results can be found in the folder `MNIST` and `CIFAR10` respectively. Further we also provide trained models for each of the adversarial training methods in the sub-folder `Selected` for the two datasets.

## Dependencies
The repository is written using `python 3.6`. To install dependencies run the command:

`pip install -r requirements.txt`


## Robustness on MNIST Dataset
|   |P∞ | P_2	|P_1	|B-ABS | ABS | Worst-PGD | PGD Aug | MSD |
| ---------| --------- | --------- | --------- | --------- |  --------- | --------- | --------- | --------- | 
| **Clean** | 99.1\% | 99.2\% | 99.3\% | 99\% | 99\% | 98.6\% | 99.1\%  |98.3\% |
| **ℓ∞ attacks (ϵ=0.3)**  | 90.3\% | 0.4\% | 0.0\% | 77\% |   8\% | 51.0\% | 65.2\% | 62.7\% |
| **ℓ2 attacks (ϵ=2.0)**  |13.6\% | 69.2\% | 38.5\% | 39\% | 80\% | 61.9\% | 60.1\% | 67.9\% |
| **ℓ1 attacks (ϵ=10)**   |4.2\% | 43.4\% | 70.0\% | 82\% | 78\% | 52.6\% | 39.2\% | 65.0\% |
| **All Attacks**         |3.7\% | 0.4\% | 0.0\% | 39\% |   8\% | 42.1\% | 34.9\% | **58.4\%**  |

**Note:** All attacks are performed with 10 random restarts on the first 1000 test examples.

## Robustness on CIFAR10 Dataset

|   |P∞ | P_2	|P_1	| Worst-PGD | PGD Aug | MSD |
| ---------| --------- | --------- | --------- | --------- |  --------- | --------- | 
| **Clean** | 83.3\% | 90.2\% | 73.3\% | 81.0\% | 84.6\% | 81.7\%|
| **ℓ∞ attacks (ϵ=0.03)**  | 50.7\% | 28.3\% | 0.2\% | 44.9\% | 42.5\% | 47.6\% |
| **ℓ2 attacks (ϵ=0.5)**  |57.3\% | 61.6\% | 0.0\% | 61.7\% | 65.0\% | 64.3\% |
| **ℓ1 attacks (ϵ=12)**   |16.0\% | 46.6\% | 7.9\% | 39.4\% | 54.0\% | 53.4\% |
| **All Attacks**         |15.6\% | 27.5\% | 0.0\% | 34.9\% | 40.6\% | **46.1\%**  |

**Note:** All attacks are performed with 10 random restarts on the first 1000 test examples.   


## How can I cite this work?
```
@inproceedings{maini2020adversarial,
	title={Adversarial Robustness Against the Union of Multiple Perturbation Models}, 
	author={Pratyush Maini and Eric Wong and J. Zico Kolter},
	booktitle={International Conference on Machine Learning},
	year={2020},
	url = "https://arxiv.org/abs/1909.04068"
}
```