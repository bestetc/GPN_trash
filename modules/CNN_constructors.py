import torch
from torch import nn



model = CNN_constructors.ResNet_like(layers=resnet_layers, bottleneck=bottleneck, num_classes=num_classes)
class ResNet_like(nn.Module):
    """ Compile ResNet-like models based on the parameters.
    
    
    Parameters
    ----------
    layers: list
        List with quantity of ResNet Block with separable it.
    num_classes: int,
        Class quantity in dataset.
    bottleneck: bool,
        Set contructor to use bottleneck blocks.
    resnet_type: str, default='A'
        Block type using in construction.
        Could be 'A', 'B', 'C' or 'D'
    activation: str, default='relu'
        Set activation function in ResNetBlock.
        Could be 'relu', 'sigmoid' or 'swish'.
    
    Raises
    ------
    ValueError
        If resnet_type have unknown value
        
    See Also
    --------    
        [1]_Bag of Tricks for Image Classification with Convolutional Neural Networks.
    Part 4.1

    """
    def __init__(self, 
                 layers, 
                 num_classes,
                 bottleneck,
                 resnet_type='A',
                 activation='relu'
                 ):
        
        super(ResNet_like, self).__init__()
        
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
        
        self.bottleneck_block_types = {
            'A': CNN_blocks.ResNet_A_Bottleneck_Block,
            'B': CNN_blocks.ResNet_B_Bottleneck_Block,
            'C': CNN_blocks.ResNet_A_Bottleneck_Block,
            'D': CNN_blocks.ResNet_D_Bottleneck_Block
        }
        self.normal_block_types = {
            'A': CNN_blocks.ResNet_A_Block,
            'B': CNN_blocks.ResNet_B_Block,
            'C': CNN_blocks.ResNet_A_Block,
            'D': CNN_blocks.ResNet_D_Block
        }
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
                    if resnet_type
                        self.body.add_module(name='block_%d_%d'%(num+2,block+1), 
                                             module=self.bottleneck_block_types[resnet_type](
                                                 num+2, 
                                                 downsample=downsample,
                                                 activation=activation))
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
                                         module=self.normal_block_types[resnet_type](
                                             num+2, 
                                             downsample=downsample,
                                             activation=activation))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
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
    