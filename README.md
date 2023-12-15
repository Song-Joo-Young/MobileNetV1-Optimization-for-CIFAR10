# MobileNetV1-Optimization-for-CIFAR10

### Overview
This repository showcases my work on optimizing MobileNetV1 for the CIFAR10 dataset. The aim was to enhance the performance of this lightweight deep learning model while maintaining its efficiency, making it more suitable for applications in environments with limited computational resources.

### Project Introduction
MobileNetV1, known for its efficiency and portability, is a popular choice for mobile and edge computing. However, when dealing with specific datasets like CIFAR10, there's room for optimization to achieve better accuracy and efficiency. This project focuses on fine-tuning and optimizing MobileNetV1 specifically for the CIFAR10 dataset, which consists of 60,000 32x32 color images in 10 classes.

### Modifications and Optimizations
* Architecture Adjustments: Minor changes in the network architecture to better fit the CIFAR10 dataset.
* Hyperparameter Tuning: Extensive experimentation with learning rates, batch sizes, and other hyperparameters.
* Data Augmentation Techniques: Implementing various data augmentation methods to improve model generalization.
* Regularization Strategies: Applying dropout and other regularization techniques to prevent overfitting.

#### Reference : 
* MobileNetV1 paper : [https://arxiv.org/abs/1704.04861]
* Mix-up augmentation : Facebookreseach mixup-cifar10 [https://github.com/facebookresearch/mixup-cifar10]
* Pruning : Towards Any Structural Pruning [https://github.com/VainF/Torch-Pruning]
