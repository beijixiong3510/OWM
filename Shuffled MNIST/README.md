# OWM
Code for **Shuffled MNIST** in paper *[Continual Learning of Context-dependent Processing in Neural Networks](https://arxiv.org/abs/1810.01256)*

## Requirements:

- Linux: Ubuntu 16.04

- cuda9.0 & cudnn6.0

- Python 3.5.4

- torch 0.3.0 (pytorch)

- torchvision 0.2.0

- numpy 1.15.1

- scipy 1.0.0

## How to run the code

### Code for [784-800-800-10] Layers three Shuffled MNIST (98.36%)

```
python run_shuffled_3_mnist_3Layers.py
```

### Code for [784-800-800-10] Layers ten Shuffled MNIST (97.63%)

```
python run_shuffled_10_mnist_3Layers.py
```

### Code for [784-2000-2000-10] Layers ten Shuffled MNIST (97.86%)

```
python run_shuffled_10_mnist_3Layers_2000.py
```

### Code for [784-100-10] Layers ten Shuffled MNIST (95.21%)

```
python run_shuffled_10_mnist_2Layers.py
```


