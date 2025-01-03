# Neural Golfing ⛳
**Neural Golfing:** Generalising functions with the fewest number of neural network parameters

## Goal
Get at least 98% accuracy on the MNIST test set using a model with the fewest number of training parameters

## Rules
- Can only train the model on the MNIST training set
- Can use image transformations but input image size should remain the same (28x28)
- Model result must be replicatable
- Entire training process should be automated
- Optimiser weights aren't counted towards the total number of trainable parameters

## Results (After pruning)
| Number of parameters | highest accuracy | Epochs |
| -------------------- | ---------------- | ------ |
| 575                  | 98.00%           | 20     |

## Motivation
The [EfficientNet paper](https://arxiv.org/abs/1905.11946) taught us that simply scalling neural networks isn't the most efficient way to generalise to a function and was able to greatly scale down model sizes and outperform state of the art models. Neural golfing challenges are hopefully a fun method of demonstrating the importance of compressed model sizes.

PyTorch's [MNIST example](https://github.com/pytorch/examples/blob/main/mnist/main.py) produces a model with 1,199,882 parameters that is able to get an accuracy >99%, The current most compressed model in this repo is able to achieve 1% below that accuracy with only 672 parameters.
