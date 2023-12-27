# Pytorch MobileNetV1 Optimization for CIFAR10

### Overview
This repository showcases my work on optimizing MobileNetV1 for the CIFAR10 dataset. The aim was to enhance the performance of this lightweight deep learning model while maintaining its efficiency, making it more suitable for applications in environments with limited computational resources.
* If you have questions about this repository, please send an e-mail to me (songsite123@naver.com) or make an issue.

### Project Introduction
MobileNetV1, known for its efficiency and portability, is a popular choice for mobile and edge computing. However, when dealing with specific datasets like CIFAR10, there's room for optimization to achieve better accuracy and efficiency. This project focuses on fine-tuning and optimizing MobileNetV1 specifically for the CIFAR10 dataset, which consists of 60,000 32x32 color images in 10 classes.

### Modifications and Optimizations
* Architecture Adjustments: Minor changes in the network architecture to better fit the CIFAR10 dataset.
* Hyperparameter Tuning: Extensive experimentation with learning rates, batch sizes, and other hyperparameters.
* Data Augmentation Techniques: Implementing various data augmentation methods to improve model generalization.
* Regularization Strategies: Applying dropout and other regularization techniques to prevent overfitting.

### Experiment Settings
The limitation of the whole training time is 3,600 sec.
#### Baseline MoblileNetV1
* The baseline model used in this repository follows the setting used in [kuangliu github pyorch-cifar](https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenet.py).
* Training batch size: 100
* weight_decay = 0.0005
* epoch = 100
* Learning rate : 0.03
* Training time : 3484.9894 sec
* optimizer : SGD
* Data augmentation
  *       transforms.Pad(4),
          transforms.RandomHorizontalFlip(),
          transforms.RandomCrop(32),
          transforms.ToTensor()
      
#### CustomMobileNet
* Training batch size: 500
* weight_decay = 0.0005
* epoch = 105
* Learning rate : 0.03
* Training time : 3544.5987 sec
* optimizer : SGD
* Data augmentation
  *       transforms.Pad(4),
          transforms.RandomHorizontalFlip(),
          transforms.RandomCrop(32),
          transforms.ToTensor()

### Result
| Model             | Test set Accuracy (%) | FLOPS      | Total params | Total Size (MB) | pass size (MB) | Params size (MB) | epoch | Training time (sec) | Train batch size |
|-------------------|-----------------------|------------|--------------|-----------------|----------------|------------------|-------|----------------------|------------------|
| Baseline MobileNetV1       | 84.26                 | 48,412,672 | 3,217,226    | 20.25           | 7.97           | 12.27            | 100   | 3484.99              | 100              |
| MobileNetV2       | 87.35                 | 319,015,424| 2,237,770    | 76.6            | 68.05          | 8.54             | 25    | 3439.57              | 100              |
| CustomMobileNet   | 91.81                 | 28,374,016 | 1,001,642    | 14.41           | 10.58          | 3.82             | 105   | 3544.5987            | 500              |


#### Reference : 
* MobileNetV1 paper : [https://arxiv.org/abs/1704.04861]
* Mixup augmenataion : [ Facebookreseach mixup-cifar10](https://github.com/facebookresearch/mixup-cifar10)
* Mix-up augmentation : Facebookreseach mixup-cifar10 [https://github.com/facebookresearch/mixup-cifar10]
* Pruning : Towards Any Structural Pruning [https://github.com/VainF/Torch-Pruning]
