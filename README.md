# ResNet-FT

A finetune examle by chx.

## Experiment Setnp
```bash
cd src/
python main.py
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