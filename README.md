# Adversarial Robustness Against the Union of Multiple Perturbation Models

Repository for the paper [Adversarial Robustness Against the Union of Multiple Perturbation Models](https://arxiv.org/abs/1909.04068) by [Pratyush Maini](https://github.com/pratyush911), [Eric Wong](https://riceric22.github.io) and [Zico Kolter](http://zicokolter.com)

## What is robustness against a union of multiple perturbation models?
While there has been a significant body of work that has focussed on creating classifiers that are robust to adversarial perturbations within a specified ℓp ball, the majority of defences have been restricted to a particular norm. In this work we focus on developing models that are robust against multiple ℓp balls simultaneously, namely ℓ∞, ℓ2, and ℓ1 balls.


## What does this work provide?
We show that it is indeed possible to adversarially train a robust model against a union of norm-bounded attacks, by using a natural generalization of the standard PGD-based procedure for adversarial training to multiple threat models. With this approach, we are able to train standard architectures which are robust against ℓ∞, ℓ2, and ℓ1 attacks, outperforming past approaches on the MNIST dataset and providing the first CIFAR10 network trained to be simultaneously robust against (ℓ∞,ℓ2,ℓ1) threat models, which achieves adversarial accuracy rates of (47.6%,64.8%,53.4%) for (ℓ∞,ℓ2,ℓ1) perturbations with radius ϵ=(0.03,0.5,12).

## What does this reposiory contain?
Code for training and evaluating all the experiments that support the aforementioned paper are provided in this repository. 
The instructions for reproducing the results can be found in the folder `MNIST` and `CIFAR10` respectively.




