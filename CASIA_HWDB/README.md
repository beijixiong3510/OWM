## CASIA-HWDB 1.1
Code for **CASIA-HWDB** in paper *[Continual Learning of Context-dependent Processing in Neural Networks](https://arxiv.org/abs/1810.01256)*

For the [Chinese character recognition](http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html) task, in total there are 3755 characters forming the level I vocabulary, which constitutes more than 99% of the usage frequency in written Chinese literature.

## Requirements:

- Linux: Ubuntu 16.04

- cuda9.0 & cudnn6.0

- Python 3.5.4

- torch 0.3.0 (pytorch)

- torchvision 0.2.0

- numpy 1.15.1

- scipy 1.0.0

## Instructions
1. Download and deal with the data in http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html
2. To train a model, run celebA_main_torch.py ([ResNet18](https://github.com/beijixiong3510/OWM/tree/master/CASIA_HWDB/CHW_ResNet18)) to extract features of all data
3. Use the [**CHW_OWM.py**](https://github.com/beijixiong3510/OWM/blob/master/CASIA_HWDB/CHW_OWM.py) to reproduce the result
