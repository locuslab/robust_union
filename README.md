# Adversarial Robustness Against the Union of Multiple Perturbation Models

Repository for the paper [Adversarial Robustness Against the Union of Multiple Perturbation Models](https://arxiv.org/abs/1909.04068) by [Pratyush Maini](https://github.com/pratyush911), [Eric Wong](https://riceric22.github.io) and [Zico Kolter](http://zicokolter.com)

## What is robustness against a union of multiple perturbation models?
While there has been a significant body of work that has focussed on creating classifiers that are robust to adversarial perturbations within a specified ℓp ball, the majority of defences have been restricted to a particular norm. In this work we focus on developing models that are robust against multiple ℓp balls simultaneously, namely ℓ∞, ℓ2, and ℓ1 balls.


## What does this work provide?
We show that it is indeed possible to adversarially train a robust model against a union of norm-bounded attacks, by using a natural generalization of the standard PGD-based procedure for adversarial training to multiple threat models. With this approach, we are able to train standard architectures which are robust against ℓ∞, ℓ2, and ℓ1 attacks, outperforming past approaches on the MNIST dataset and providing the first CIFAR10 network trained to be simultaneously robust against (ℓ∞,ℓ2,ℓ1) threat models, which achieves adversarial accuracy rates of (47.6%,64.8%,53.4%) for (ℓ∞,ℓ2,ℓ1) perturbations with radius ϵ=(0.03,0.5,12).

## What does this repository contain?
Code for training and evaluating all the experiments that support the aforementioned paper are provided in this repository. 
The instructions for reproducing the results can be found in the folder `MNIST` and `CIFAR10` respectively.

## Robustness on MNIST Dataset
|   |P∞ | P_2	|P_1	|B-ABS | ABS | Worst-PGD | PGD Aug | MSD |
| ---------| --------- | --------- | --------- | --------- |  --------- | --------- | --------- | --------- | 
| **Clean** | 99.1\% | 99.4\% | 98.9\% | 99\% | 99\% | 98.9\% | 99.1\%  |98.0\% |
| **ℓ∞ attacks (ϵ=0.3)**  | 90.3\% | 0.4\% | 0.0\% | 77\% |   8\% | 68.4\% | 83.7\% | 63.7\% |
| **ℓ2 attacks (ϵ=1.5)**  |46.4\% | 87.0\% | 70.7\% | 39\% | 80\% | 82.6\% | 76.2\% | 82.7\% |
| **ℓ1 attacks (ϵ=12)**   |1.4\% | 43.4\% | 71.8\% | 82\% | 78\% | 54.6\% | 15.6\% | 62.3\% |
| **All Attacks**         |1.4\% | 0.4\% | 0.0\% | 39\% |   8\% | 53.7\% | 15.6\% | **58.7\%**  |

**Note:** All attacks are performed with 10 random restarts on the first 1000 test examples.

## Robustness on CIFAR10 Dataset

|   |P∞ | P_2	|P_1	| Worst-PGD | PGD Aug | MSD |
| ---------| --------- | --------- | --------- | --------- |  --------- | --------- | 
| **Clean** | 83.3\% | 90.2\% | 73.3\% | 81.0\% | 84.6\% | 81.7\%|
| **ℓ∞ attacks (ϵ=0.03)**  | 50.7\% | 28.3\% | 0.2\% | 44.9\% | 42.5\% | 47.6\% |
| **ℓ2 attacks (ϵ=0.5)**  |58.2\% | 61.6\% | 0.0\% | 62.1\% | 65.3\% | 64.8\% |
| **ℓ1 attacks (ϵ=12)**   |16.0\% | 46.6\% | 7.9\% | 39.4\% | 54.0\% | 53.4\% |
| **All Attacks**         |15.6\% | 25.2\% | 0.0\% | 34.9\% | 40.6\% | **46.1\%**  |

**Note:** All attacks are performed with 10 random restarts on the first 1000 test examples.
