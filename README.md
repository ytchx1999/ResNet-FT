# ResNet-FT

A finetune examle by chx.

## Experiment Setnp
```bash
cd src/
python main.py
```
## Structure of ResNet-18

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 112, 112]           9,408
       BatchNorm2d-2         [-1, 64, 112, 112]             128
              ReLU-3         [-1, 64, 112, 112]               0
         MaxPool2d-4           [-1, 64, 56, 56]               0
            Conv2d-5           [-1, 64, 56, 56]          36,864
       BatchNorm2d-6           [-1, 64, 56, 56]             128
              ReLU-7           [-1, 64, 56, 56]               0
            Conv2d-8           [-1, 64, 56, 56]          36,864
       BatchNorm2d-9           [-1, 64, 56, 56]             128
             ReLU-10           [-1, 64, 56, 56]               0
       BasicBlock-11           [-1, 64, 56, 56]               0
           Conv2d-12           [-1, 64, 56, 56]          36,864
      BatchNorm2d-13           [-1, 64, 56, 56]             128
             ReLU-14           [-1, 64, 56, 56]               0
           Conv2d-15           [-1, 64, 56, 56]          36,864
      BatchNorm2d-16           [-1, 64, 56, 56]             128
             ReLU-17           [-1, 64, 56, 56]               0
       BasicBlock-18           [-1, 64, 56, 56]               0
           Conv2d-19          [-1, 128, 28, 28]          73,728
      BatchNorm2d-20          [-1, 128, 28, 28]             256
             ReLU-21          [-1, 128, 28, 28]               0
           Conv2d-22          [-1, 128, 28, 28]         147,456
      BatchNorm2d-23          [-1, 128, 28, 28]             256
           Conv2d-24          [-1, 128, 28, 28]           8,192
      BatchNorm2d-25          [-1, 128, 28, 28]             256
             ReLU-26          [-1, 128, 28, 28]               0
       BasicBlock-27          [-1, 128, 28, 28]               0
           Conv2d-28          [-1, 128, 28, 28]         147,456
      BatchNorm2d-29          [-1, 128, 28, 28]             256
             ReLU-30          [-1, 128, 28, 28]               0
           Conv2d-31          [-1, 128, 28, 28]         147,456
      BatchNorm2d-32          [-1, 128, 28, 28]             256
             ReLU-33          [-1, 128, 28, 28]               0
       BasicBlock-34          [-1, 128, 28, 28]               0
           Conv2d-35          [-1, 256, 14, 14]         294,912
      BatchNorm2d-36          [-1, 256, 14, 14]             512
             ReLU-37          [-1, 256, 14, 14]               0
           Conv2d-38          [-1, 256, 14, 14]         589,824
      BatchNorm2d-39          [-1, 256, 14, 14]             512
           Conv2d-40          [-1, 256, 14, 14]          32,768
      BatchNorm2d-41          [-1, 256, 14, 14]             512
             ReLU-42          [-1, 256, 14, 14]               0
       BasicBlock-43          [-1, 256, 14, 14]               0
           Conv2d-44          [-1, 256, 14, 14]         589,824
      BatchNorm2d-45          [-1, 256, 14, 14]             512
             ReLU-46          [-1, 256, 14, 14]               0
           Conv2d-47          [-1, 256, 14, 14]         589,824
      BatchNorm2d-48          [-1, 256, 14, 14]             512
             ReLU-49          [-1, 256, 14, 14]               0
       BasicBlock-50          [-1, 256, 14, 14]               0
           Conv2d-51            [-1, 512, 7, 7]       1,179,648
      BatchNorm2d-52            [-1, 512, 7, 7]           1,024
             ReLU-53            [-1, 512, 7, 7]               0
           Conv2d-54            [-1, 512, 7, 7]       2,359,296
      BatchNorm2d-55            [-1, 512, 7, 7]           1,024
           Conv2d-56            [-1, 512, 7, 7]         131,072
      BatchNorm2d-57            [-1, 512, 7, 7]           1,024
             ReLU-58            [-1, 512, 7, 7]               0
       BasicBlock-59            [-1, 512, 7, 7]               0
           Conv2d-60            [-1, 512, 7, 7]       2,359,296
      BatchNorm2d-61            [-1, 512, 7, 7]           1,024
             ReLU-62            [-1, 512, 7, 7]               0
           Conv2d-63            [-1, 512, 7, 7]       2,359,296
      BatchNorm2d-64            [-1, 512, 7, 7]           1,024
             ReLU-65            [-1, 512, 7, 7]               0
       BasicBlock-66            [-1, 512, 7, 7]               0
AdaptiveAvgPool2d-67            [-1, 512, 1, 1]               0
           Linear-68                    [-1, 2]           1,026
           ResNet-69                    [-1, 2]               0
================================================================
Total params: 11,177,538
Trainable params: 11,177,538
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 62.79
Params size (MB): 42.64
Estimated Total Size (MB): 106.00
----------------------------------------------------------------
```
## Results

```bash
epoch: 00, train_loss: 0.0370, train_acc: 0.6816, val_loss, 0.0208, val_acc, 0.8954 
epoch: 01, train_loss: 0.0243, train_acc: 0.8531, val_loss, 0.0129, val_acc, 0.9477 
epoch: 02, train_loss: 0.0158, train_acc: 0.9224, val_loss, 0.0140, val_acc, 0.9412 
epoch: 03, train_loss: 0.0144, train_acc: 0.9143, val_loss, 0.0115, val_acc, 0.9542 
epoch: 04, train_loss: 0.0108, train_acc: 0.9306, val_loss, 0.0131, val_acc, 0.9346 
epoch: 05, train_loss: 0.0074, train_acc: 0.9551, val_loss, 0.0114, val_acc, 0.9346 
epoch: 06, train_loss: 0.0103, train_acc: 0.9347, val_loss, 0.0147, val_acc, 0.9281 
epoch: 07, train_loss: 0.0093, train_acc: 0.9469, val_loss, 0.0170, val_acc, 0.8954 
epoch: 08, train_loss: 0.0137, train_acc: 0.9224, val_loss, 0.0100, val_acc, 0.9542 
epoch: 09, train_loss: 0.0075, train_acc: 0.9510, val_loss, 0.0119, val_acc, 0.9412 
epoch: 10, train_loss: 0.0068, train_acc: 0.9510, val_loss, 0.0091, val_acc, 0.9412 
epoch: 11, train_loss: 0.0068, train_acc: 0.9429, val_loss, 0.0127, val_acc, 0.9216 
epoch: 12, train_loss: 0.0085, train_acc: 0.9510, val_loss, 0.0132, val_acc, 0.9346 
epoch: 13, train_loss: 0.0090, train_acc: 0.9551, val_loss, 0.0151, val_acc, 0.9216 
epoch: 14, train_loss: 0.0093, train_acc: 0.9551, val_loss, 0.0132, val_acc, 0.9281 
epoch: 15, train_loss: 0.0050, train_acc: 0.9673, val_loss, 0.0149, val_acc, 0.9346 
epoch: 16, train_loss: 0.0050, train_acc: 0.9755, val_loss, 0.0118, val_acc, 0.9412 
epoch: 17, train_loss: 0.0051, train_acc: 0.9714, val_loss, 0.0115, val_acc, 0.9412 
epoch: 18, train_loss: 0.0073, train_acc: 0.9673, val_loss, 0.0113, val_acc, 0.9412 
epoch: 19, train_loss: 0.0058, train_acc: 0.9714, val_loss, 0.0114, val_acc, 0.9281 
```

## Tree

```bash
.
├── README.md
├── data
│   ├── hymenoptera_data
│   │   ├── train
│   │   │   ├── ants
│   │   │   │   ├── 0013035.jpg
│   │   │   │   ├── 1030023514_aad5c608f9.jpg
│   │   │   │   ├── ...
│   │   │   └── bees
│   │   │       ├── 1092977343_cb42b38d62.jpg
│   │   │       ├── 1093831624_fb5fbe2308.jpg
│   │   │       └── ...
│   │   └── val
│   │       ├── ants
│   │       │   ├── 10308379_1b6c72e180.jpg
│   │       │   ├── 1053149811_f62a3410d3.jpg
│   │       │   └── ...
│   │       └── bees
│   │           ├── 1032546534_06907fe3b3.jpg
│   │           ├── 10870992_eebeeb3a12.jpg
│   │           └── ...
│   └── hymenoptera_data.zip
└── src
    ├── __init__.py
    ├── main.py
    ├── models
    │   ├── __init__.py
    │   ├── __pycache__
    │   │   ├── __init__.cpython-37.pyc
    │   │   └── model.cpython-37.pyc
    │   ├── model.py
    │   └── resnet18-5c106cde.pth
    └── utils
        ├── __init__.py
        ├── __pycache__
        │   ├── __init__.cpython-37.pyc
        │   └── dataset.cpython-37.pyc
        ├── config.py
        └── dataset.py

```