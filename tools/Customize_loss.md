## Add new loss
``` Python
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES

@LOSSES.register_module()
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, loss_name='iou_loss'):
        super(IoULoss, self).__init__()
        self.weight = weight
        self._loss_name = loss_name

    def forward(self, inputs, targets, smooth=1, weight=None, ignore_index=None):
        num_classes = inputs.size(1)
        loss = 0

        for cls in range(num_classes):
            _targets = (targets==cls).long()
            _inputs = inputs[:,cls,...]
            #comment out if your model contains a sigmoid or equivalent activation layer
            _inputs = F.sigmoid(_inputs)       
            
            #flatten label and prediction tensors
            _inputs = _inputs.view(-1)
            _targets = _targets.view(-1)
            
            #intersection is equivalent to True Positive count
            #union is the mutually inclusive area of all labels & predictions 
            intersection = (_inputs * _targets).sum()
            total = (_inputs + _targets).sum()
            union = total - intersection 
            
            IoU = (intersection + smooth)/(union + smooth)
            loss += 1-IoU
        return loss
    
    @property
    def loss_name(self):
        return self._loss_name
```
>mmseg/models/losses 에 my_loss.py를 추가합니다.

>위 예제에서 처럼 사용하고자 하는 class에 **@LOSSES.register_module** 을 넣고 **@property**밑에 **loss_name** 함수를 정의합니다.

>forward 함수에 weight, ignore_index 인자가 반드시 있어야 합니다.

>  \_\_init\_\_.py에 from .my_loss import my_class 이런식으로 추가합니다.
밑에 \_\_all\_\_에도 loss 이름에 맞게 추가해줍니다.


> 이제 모델에서 적용해줍니다.

