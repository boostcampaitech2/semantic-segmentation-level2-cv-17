# 1. Segmentation Models Pytorch
## 🛠 Installation
PyPI version:

```$ pip install segmentation-models-pytorch```

Latest version from source:

```$ pip install git+https://github.com/qubvel/segmentation_models.pytorch```

## ⏳ Quick start
### train
``` python3 smp_train.py```

### inference
```python3 smp_inference.py```

## Advanced Example
### train
```python train.py --batch_size 16 --seed 1995 ...```
### Train Arguments
|Argument|DataType|Default|Help|
|---|---|---|---|
| seed            | int         | 21             | random seed                   |
| epochs          | int         | 20               | number of epochs to train     |
| batch_size      | int         | 8             | input batch size for training |
| model           | str         | Unet  | model type                    |
| encoder_name    | str         | hr_w18  | encoder type                    |
| optimizer       | str         | Adam              | optimizer type                |
| lr              | float       | 1e-4             | learning rate                 |
| name            | str         | -     | dir name for trained model    |
| data_dir        | str         | _                | image data path               |
| model_dir       | str         | _                | model saving path             |

---

# 2. Torchvision models
## 📑 Document
[Semantic Segmentation](https://pytorch.org/vision/stable/models.html#semantic-segmentation)

## ⏳ Quick start
### train
``` python3 torchvision_train.py```

## 추가할 사항
+ wandb 연동
+ ~~torchvision train~~  
+ torchvision inference
