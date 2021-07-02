""" Module contain a few constructors for CNN.

"""

import torch
from torch import nn

from .CNN_blocks import ResNetNormalBlock, ResNetBottleneckBlock

class ResNetLike(nn.Module):
    """ Compile ResNet-like neural networks.   
    
    Compile any kind of XResNet architecrutes. Standart or custom.
    Be carefull to use a lot of the layers of blocks. Each block layer are double
    channels used in it. Computation cost may be to high.
    
    Parameters
    ----------
    layers: iterable with int (list, tuples etc)
        List with quantity of ResNet blocks separated by the layers of blocks.
    num_classes: int,
        Class quantity in dataset.
    bottleneck: bool,
        Set constructor to use bottleneck blocks.
    resnet_type: str, optional
        Block type using in construction.
        Could be 'A', 'B', 'C' or 'D'
    activation: str, optional
        Set activation function used in ResNet block.
        Could be 'relu', 'sigmoid' or 'swish'.
    self_attention: bool, optional
        add self attention layer at the end of the first blocks group.
    
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
                 activation='relu',
                 self_attention=False,
                 blur_pool=False
                 ):
        
        self.layers = layers
        self.num_classes = num_classes
        self.bottleneck = bottleneck
        self.resnet_type = resnet_type
        self.activation = activation
        self.self_attention = self_attention
        self.blur_pool = blur_pool

        
        super().__init__()
        
        if resnet_type in ['C', 'D']:
            self.first = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                
        elif resnet_type in ['A', 'B']:
            self.first = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            raise ValueError('Unknown resnet_type value')
        
        self.body = nn.Sequential()
        
        b = 4 if bottleneck else 1 # channels multiplier
        for num, layer in enumerate(layers):
            for block in range(layer):
                use_attention = False
                if num == 0 and block == layers[0] and self_attention:
                    use_attention = True
                if block == 0 and num == 0:
                    in_channels = 64
                    out_channels = 64 * b
                    downsample = False
                elif block == 0 and num > 0:
                    in_channels = 64 * (2**(num-1)) * b
                    out_channels = 64 * (2**num) * b
                    downsample = True
                else:
                    in_channels = 64 * (2**num) * b
                    out_channels = 64 * (2**num) * b
                    downsample = False
                cnn_block_module = ResNetBottleneckBlock if bottleneck else ResNetNormalBlock
                self.body.add_module(name='block_%d_%d'%(num+2, block+1), # naming for corresponding with ResNet paper
                                     module=cnn_block_module(
                                         in_channels,
                                         out_channels,
                                         downsample=downsample,
                                         activation=activation,
                                         block_type=resnet_type,
                                         use_attention=use_attention
                                     ))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_input = 32*(2**(len(layers))) * b
        self.linear = nn.Linear(self.linear_input, num_classes)
        
    def forward(self, x):

        x = self.first(x)
        x = self.body(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
       
        return x
