""" Module contain a few constructors for CNN.

"""

import torch
from torch import nn

from modules import CNN_blocks

class ResNetLike(nn.Module):
    """ Compile ResNet-like neural networks.   
    
    Compile any kind of XResNet architecrutes. Standart or custom.
    Be carefull to use a lot of the layers of blocks. Each block layer are double
    channels used in it. Computation cost may be to high.
    
    Parameters
    ----------
    layers: list
        List with quantity of ResNet block separated by the layer of blocks.
    num_classes: int,
        Class quantity in dataset.
    bottleneck: bool,
        Set contructor to use bottleneck blocks.
    resnet_type: str, optional
        Block type using in construction.
        Could be 'A', 'B', 'C' or 'D'
    activation: str, optional
        Set activation function used in ResNet block.
        Could be 'relu', 'sigmoid' or 'swish'.
    
    Raises
    ------
    ValueError
        If resnet_type have unknown value
        
    Notes
    -----   
        [1]_Bag of Tricks for Image Classification with Convolutional Neural Networks. Part 4.
        https://arxiv.org/pdf/1812.01187.pdf
        
    Examples
    --------
        ResNetLike([3, 4, 6, 3], 1000, False): 
            create standart ResNet-34 for ImageNet dataset
        ResNetLike([3, 4, 6, 3], 1000, True): 
            create standart ResNet-50 for ImageNet dataset
        
        ResNetLike([2, 2, 2, 2], 10, False, resnet_type='D'):
            create ResNetD-18 for Imagenette dataset
            
        ResNetLike([3, 4, 23, 3], 10, False, resnet_type='B', activation='swish'):
            create ResNetB-101 for Imagenette dataset with swish activation function

    """
    def __init__(self, 
                 layers, 
                 num_classes,
                 bottleneck,
                 resnet_type='A',
                 activation='relu'
                 ):
        
        super().__init__()
        
        if resnet_type in ['C', 'D']:
            self.first = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64))
        elif resnet_type in ['A', 'B']:
            self.first = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            raise ValueError('Unknown resnet_type value')
        
        self.body = nn.Sequential()
        if bottleneck:
            for num, layer in enumerate(layers):
                for block in range(layer):
                    if block == 0  and num < len(layers) - 1:
                        downsample = 1
                    elif block == 0 and num == len(layers) - 1:
                        downsample = -1
                    elif block != 0:  
                        downsample = 0
                    self.body.add_module(name='block_%d_%d'%(num+2,block+1), 
                                         module=CNN_blocks.ResNetBottleneckBlock(
                                             num+2, 
                                             downsample=downsample,
                                             activation=activation,
                                             block_type=resnet_type))
        elif not bottleneck:
            for num, layer in enumerate(layers):
                for block in range(layer):
                    if block == 0  and num < len(layers) - 1:
                        downsample = 1
                    elif block == 0 and num == len(layers) - 1:
                        downsample = -1
                    elif block != 0:  
                        downsample = 0
                    self.body.add_module(name='block_%d_%d'%(num+2,block+1), 
                                         module=CNN_blocks.ResNetNormalBlock(
                                             num+2, 
                                             downsample=downsample,
                                             activation=activation,
                                             block_type=resnet_type))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        m = 4 if bottleneck else 1
        self.linear_input = 32*(2**(len(layers))) * m
        self.linear = nn.Linear(self.linear_input, num_classes)
        
    def forward(self, x):

        x = self.first(x)
        x = self.body(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
       
        return x
    