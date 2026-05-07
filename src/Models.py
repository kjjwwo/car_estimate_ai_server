import torch
from torch import nn, optim
import segmentation_models_pytorch as smp


class Unet(nn.Module):
    def __init__(self, num_classes,encoder,pre_weight):         
        super().__init__()                                      
        self.model = smp.Unet( classes = num_classes,           # Unet 모델을 사용
                              encoder_name=encoder,             # encoder는 resnet34
                              encoder_weights=pre_weight,       # encoder_weights는 imagenet
                              in_channels=3)                    
    
    def forward(self, x):
        y = self.model(x)
        encoder_weights = "imagenet"
        return y

    