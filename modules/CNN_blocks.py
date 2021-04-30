from torch import nn

def conv1x1(in_channels, out_channels, stride=1, padding=0):
    """ Return convolution layer with kernel_size is 1 
    
    See more torch.nn.Conv2d
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding)

def conv3x3(in_channels, out_channels, stride=1, padding=1):
    """ Make convolution layer with kernel_size is 3 
        
    See more torch.nn.Conv2d
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)

class ResNet_A_Block(nn.Module):
    """ Create Normal ResNet Block 
    
    Create ResNet block with two convolutional layers with skip connection.
    All convolutional layer have normalize with BatchNorm layer.
    Downsampling realize with three mode: 
        1 - downsampling decrease H and W of tensor in 2 times and increase channels in 2 times
        -1 - downsampling dont change H and W and increase channels in 2 times
        0 - no downsampling  
    
    Parameters
    ----------
    num_layer: int
        block serial number. Used for channels calc.
        standart ResNet start to use ResNet block with second block (num_layer=2)
    downsample: int, optional
        set type of downsampling in ResNet Block
        should be -1, 0 or 1
    activation: str, optional
        set activation function in ResNetBlock
        should be 'relu', 'sigmoid' or 'swish'
    
    Raises
    ------
    ValueError
        If downsample have unaccaptable value.
        
    See Also
    --------
    ResNet_B_Block
    ResNet_D_Block
    ResNet_A_Bottleneck_Block
    ResNet_B_Bottleneck_Block
    ResNet_D_Bottleneck_Block

    [1]_Bag of Tricks for Image Classification with Convolutional Neural Networks. Part 4.1.
    https://arxiv.org/pdf/1812.01187.pdf
    """
    
    def __init__(self,
                 num_layer,
                 downsample=0,
                 activation='relu'
                 ):
        super(ResNet_A_Block, self).__init__()
        
        if downsample not in [-1, 0, 1]:
            raise ValueError('Downsample have unacceptable value')
        self.use_downsample = downsample
            
        if num_layer == 2 and downsample == 1:
            self.in_channels = 16*(2**num_layer)
        elif num_layer > 2 and downsample != 0:
            self.in_channels = 16*(2**(num_layer-1))
        elif downsample == 0: 
            self.in_channels = 16*(2**num_layer)
        self.out_channels = 16*(2**num_layer)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        
        if downsample == 1:
            self.downsample = nn.Sequential(
                conv1x1(self.in_channels, self.out_channels, stride=2),
                nn.BatchNorm2d(self.out_channels))
            
            self.conv1 = conv3x3(self.in_channels, self.out_channels, stride=2)
            self.in_channels = self.out_channels
            self.bn1 = nn.BatchNorm2d(self.out_channels)
        elif downsample == -1:
            self.downsample = nn.Sequential(
                conv1x1(self.in_channels, self.out_channels, stride=1),
                nn.BatchNorm2d(self.out_channels)
                )
            self.conv1 = conv3x3(self.in_channels, self.out_channels)
            self.in_channels = self.out_channels
            self.bn1 = nn.BatchNorm2d(self.out_channels)
        elif downsample == 0:
            self.conv1 = conv3x3(self.in_channels, self.out_channels)
            self.in_channels = self.out_channels
            self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv3x3(self.in_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        
    def forward(self, x):
        skip = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_downsample != 0:
            skip = self.downsample(x)

        out += skip
        out = self.activation(out)

        return out
    
class ResNet_A_Bottleneck_Block(nn.Module):
    """ Create Bottleneck ResNet Block 
    
    Create ResNet block with three convolutional layers with skip connection.
    All convolutional layer have normalize with BatchNorm layer.
    Downsampling realize with three mode: 
        1: downsampling decrease H and W of tensor in 2 times and increase channels in 2 times
        -1: downsampling dont change H and W and increase channels in 2 times
        0: no downsampling  
    
    Parameters
    ----------
    num_layer: int
        block serial number. Used for channels calc.
        standart ResNet start to use ResNet block with second block (num_layer=2)
    downsample: int, optional
        set type of downsampling in ResNet Block
        should be -1, 0 or 1
    activation: str, optional
        set activation function in ResNetBlock
        should be 'relu', 'sigmoid' or 'swish'

    See Also
    --------
    ResNet_A_Block
    ResNet_B_Block
    ResNet_D_Block
    ResNet_B_Bottleneck_Block
    ResNet_D_Bottleneck_Block
    
    [1]_Bag of Tricks for Image Classification with Convolutional Neural Networks. Part 4.1.
    https://arxiv.org/pdf/1812.01187.pdf
    """
    def __init__(self,
                 num_layer,
                 downsample=0,
                 activation='relu'
                 ):
        super(ResNet_A_Bottleneck_Block, self).__init__()
        
        if downsample not in [-1, 0, 1]:
            raise ValueError('Downsample have unacceptable value')
        self.use_downsample = downsample

        if num_layer == 2 and downsample == 1:
            self.in_channels = 16 * (2**num_layer)
        elif num_layer > 2 and downsample != 0:
            self.in_channels = 16 * (2**(num_layer - 1)) * 4
        elif downsample == 0: 
            self.in_channels = 16 * (2**num_layer) * 4
        self.out_channels = 16 * (2**num_layer)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'swish':
            self.activation = nn.SiLU()
            
        if downsample == 1:
            self.downsample = nn.Sequential(
                conv1x1(self.in_channels, self.out_channels * 4, stride=2),
                nn.BatchNorm2d(self.out_channels * 4))
            self.conv1 = conv1x1(self.in_channels, self.out_channels, stride=2)
            self.in_channels = self.out_channels
            self.bn1 = nn.BatchNorm2d(self.out_channels)
        elif downsample == -1:
            self.downsample = nn.Sequential(
                conv1x1(self.in_channels, self.out_channels * 4, stride=1),
                nn.BatchNorm2d(self.out_channels * 4))
            self.conv1 = conv1x1(self.in_channels, self.out_channels)
            self.in_channels = self.out_channels
            self.bn1 = nn.BatchNorm2d(self.out_channels)
        elif downsample == 0:
            self.conv1 = conv1x1(self.in_channels, self.out_channels)
            self.in_channels = self.out_channels
            self.bn1 = nn.BatchNorm2d(self.out_channels)
            
        self.conv2 = conv3x3(self.in_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.out_channels = 16 * (2**num_layer) * 4
        self.conv3 = conv1x1(self.in_channels, self.out_channels)
        self.bn3 = nn.BatchNorm2d(self.out_channels)
        self.in_channels = self.out_channels
        
    def forward(self, x):
        skip = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.use_downsample != 0:
            skip = self.downsample(x)

        out += skip
        out = self.activation(out)
            
        return out

class ResNet_B_Block(nn.Module):
    """ Create Normal ResNet-B Block 
    
    Create ResNet-B type block with two convolutional layers with skip connection.
    All convolutional layer have normalize with BatchNorm layer.
    Downsampling realize with three mode: 
        1 - downsampling decrease H and W of tensor in 2 times and increase channels in 2 times
        -1 - downsampling dont change H and W and increase channels in 2 times
        0 - no downsampling  
    
    Parameters
    ----------
    num_layer: int
        block serial number. Used for channels calc.
        standart ResNet start to use ResNet block with second block (num_layer=2)
    downsample: int, optional
        set type of downsampling in ResNet Block
        should be -1, 0 or 1
    activation: str, optional
        set activation function in ResNetBlock
        should be 'relu', 'sigmoid' or 'swish'
        
    See Also
    --------
    ResNet_A_Block
    ResNet_D_Block
    ResNet_A_Bottleneck_Block
    ResNet_B_Bottleneck_Block
    ResNet_D_Bottleneck_Block
        
    [1]_Bag of Tricks for Image Classification with Convolutional Neural Networks. Part 4.2.
    https://arxiv.org/pdf/1812.01187.pdf
    
    
    """
    
    def __init__(self,
                 num_layer,
                 downsample=0,
                 activation='relu'
                 ):
        super(ResNet_B_Block, self).__init__()
        
        if downsample not in [-1, 0, 1]:
            raise ValueError('Downsample have unacceptable value')
        self.use_downsample = downsample
        
        if num_layer == 2 and downsample == 1:
            self.in_channels = 16 * (2**num_layer)
        if num_layer > 2 and downsample != 0:
            self.in_channels = 16 * (2**(num_layer - 1))
        elif downsample == 0: 
            self.in_channels = 16 * (2**num_layer)
        self.out_channels = 16 * (2**num_layer)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'swish':
            self.activation = nn.SiLU()
            
        if downsample == 1:
            self.downsample = nn.Sequential(
                conv1x1(self.in_channels, self.out_channels, stride=2),
                nn.BatchNorm2d(self.out_channels))
            self.conv1 = conv3x3(self.in_channels, self.out_channels, stride=2)
        elif downsample == -1:
            self.downsample = conv1x1(self.in_channels, self.out_channels, stride=1)
            self.conv1 = conv3x3(self.in_channels, self.out_channels, stride=1)
        elif downsample == 0:
            self.conv1 = conv3x3(self.in_channels, self.out_channels, stride=1)
            
        self.in_channels = self.out_channels
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv3x3(self.in_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        
    def forward(self, x):
        skip = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_downsample != 0:
            skip = self.downsample(x)

        out += skip
        out = self.activation(out)

        return out
    
class ResNet_B_Bottleneck_Block(nn.Module):
    """ Create Bottleneck ResNet Block 
    
    Create ResNet-B type block with three convolutional layers with skip connection.
    All convolutional layer have normalize with BatchNorm layer.
    Downsampling realize with three mode: 
        1 - downsampling decrease H and W of tensor in 2 times and increase channels in 2 times
        -1 - downsampling dont change H and W and increase channels in 2 times
        0 - no downsampling  
    
    Parameters
    ----------
    num_layer: int
        block serial number. Used for channels calc.
        standart ResNet start to use ResNet block with second block (num_layer=2)
    downsample: int, optional
        set type of downsampling in ResNet Block
        should be -1, 0 or 1
    activation: str, optional
        set activation function in ResNetBlock
        should be 'relu', 'sigmoid' or 'swish'

    See Also
    --------
    ResNet_A_Block
    ResNet_B_Block
    ResNet_D_Block
    ResNet_A_Bottleneck_Block
    ResNet_D_Bottleneck_Block

    [1]_Bag of Tricks for Image Classification with Convolutional Neural Networks. Part 4.2.
    https://arxiv.org/pdf/1812.01187.pdf

    """
    
    def __init__(self,
                 num_layer,
                 downsample=0,
                 activation='relu'
                 ):
        super(ResNet_B_Bottleneck_Block, self).__init__()
        
        if downsample not in [-1, 0, 1]:
            raise ValueError('Downsample have unacceptable value')
        self.use_downsample = downsample
        
        if num_layer == 2 and downsample == 1:
            self.in_channels = 16 * (2**num_layer)
        elif num_layer > 2 and downsample != 0:
            self.in_channels = 16 * (2**(num_layer-1)) * 4
        elif downsample == 0: 
            self.in_channels = 16 * (2**num_layer) * 4
        self.out_channels = 16 * (2**num_layer)
   
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'swish':
            self.activation = nn.SiLU()
            
        if downsample == 1:
            self.downsample = nn.Sequential(
                conv1x1(self.in_channels, self.out_channels * 4, stride=2),
                nn.BatchNorm2d(self.out_channels * 4))
            self.conv1 = conv1x1(self.in_channels, self.out_channels)
            self.in_channels = self.out_channels
            self.bn1 = nn.BatchNorm2d(self.out_channels)
            self.conv2 = conv3x3(self.in_channels, self.out_channels, stride=2)
        elif downsample == -1:
            self.downsample = nn.Sequential(
                conv1x1(self.in_channels, self.out_channels * 4, stride=1),
                nn.BatchNorm2d(self.out_channels * 4))
            self.conv1 = conv1x1(self.in_channels, self.out_channels)
            self.in_channels = self.out_channels
            self.bn1 = nn.BatchNorm2d(self.out_channels)
            self.conv2 = conv3x3(self.in_channels, self.out_channels)
        elif downsample == 0:
            self.conv1 = conv1x1(self.in_channels, self.out_channels)
            self.in_channels = self.out_channels
            self.bn1 = nn.BatchNorm2d(self.out_channels)
            self.conv2 = conv3x3(self.in_channels, self.out_channels)
        
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.out_channels = 16 * (2**num_layer) * 4
        self.conv3 = conv1x1(self.in_channels, self.out_channels)
        self.bn3 = nn.BatchNorm2d(self.out_channels)
        self.in_channels = self.out_channels
        
    def forward(self, x):
        skip = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.use_downsample != 0:
            skip = self.downsample(x)

        out += skip
        out = self.activation(out)
            
        return out

    
class ResNet_D_Block(nn.Module):
    """ Create Normal ResNet-D Block 
    
    Create ResNet-D type block with two convolutional layers with skip connection.
    All convolutional layer have normalize with BatchNorm layer.
    Downsampling realize with three mode: 
        1 - downsampling decrease H and W of tensor in 2 times and increase channels in 2 times
        -1 - downsampling dont change H and W and increase channels in 2 times
        0 - no downsampling  
    
    Parameters
    ----------
    num_layer: int
        block serial number. Used for channels calc.
        standart ResNet start to use ResNet block with second block (num_layer=2)
    downsample: int, optional
        set type of downsampling in ResNet Block
        should be -1, 0 or 1
    activation: str, optional
        set activation function in ResNetBlock
        should be 'relu', 'sigmoid' or 'swish'
        
    See Also
    --------
    ResNet_A_Block
    ResNet_B_Block
    ResNet_A_Bottleneck_Block
    ResNet_B_Bottleneck_Block
    ResNet_D_Bottleneck_Block
        
    [1]_Bag of Tricks for Image Classification with Convolutional Neural Networks.
    Part 4.2
    
    """
    
    def __init__(self,
                 num_layer,
                 downsample=0,
                 activation='relu'
                 ):
        super(ResNet_D_Block, self).__init__()
        
        if downsample not in [-1, 0, 1]:
            raise ValueError('Downsample have unacceptable value')
        self.use_downsample = downsample
        
        if num_layer == 2 and downsample == 1:
            self.in_channels = 16 * (2**num_layer)
        if num_layer > 2 and downsample != 0:
            self.in_channels = 16 * (2**(num_layer - 1))
        elif downsample == 0: 
            self.in_channels = 16 * (2**num_layer)
        self.out_channels = 16 * (2**num_layer)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'swish':
            self.activation = nn.SiLU()
            
        if downsample == 1:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                conv1x1(self.in_channels, self.out_channels, stride=1),
                nn.BatchNorm2d(self.out_channels))
            self.conv1 = conv3x3(self.in_channels, self.out_channels, stride=2)
        elif downsample == -1:
            self.downsample = conv1x1(self.in_channels, self.out_channels, stride=1)
            self.conv1 = conv3x3(self.in_channels, self.out_channels, stride=1)
        elif downsample == 0:
            self.conv1 = conv3x3(self.in_channels, self.out_channels, stride=1)
            
        self.in_channels = self.out_channels
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv3x3(self.in_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        
    def forward(self, x):
        skip = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_downsample != 0:
            skip = self.downsample(x)

        out += skip
        out = self.activation(out)

        return out
    
class ResNet_D_Bottleneck_Block(nn.Module):
    """ Create Bottleneck ResNet Block 
    
    Create ResNet-D type block with three convolutional layers with skip connection.
    All convolutional layer have normalize with BatchNorm layer.
    Downsampling realize with three mode: 
        1 - downsampling decrease H and W of tensor in 2 times and increase channels in 2 times
        -1 - downsampling dont change H and W and increase channels in 2 times
        0 - no downsampling  
    
    Parameters
    ----------
    num_layer: int
        block serial number. Used for channels calc.
        standart ResNet start to use ResNet block with second block (num_layer=2)
    downsample: int, optional
        set type of downsampling in ResNet Block
        should be -1, 0 or 1
    activation: str, optional
        set activation function in ResNetBlock
        should be 'relu', 'sigmoid' or 'swish'

    See Also
    --------
    ResNet_A_Block
    ResNet_B_Block
    ResNet_D_Block
    ResNet_A_Bottleneck_Block
    ResNet_B_Bottleneck_Block

    [1]_Bag of Tricks for Image Classification with Convolutional Neural Networks. Part 4.2.
    https://arxiv.org/pdf/1812.01187.pdf
    
    """
    
    def __init__(self,
                 num_layer,
                 downsample=0,
                 activation='relu'
                 ):
        super(ResNet_D_Bottleneck_Block, self).__init__()
        
        if downsample not in [-1, 0, 1]:
            raise ValueError('Downsample have unacceptable value')
        self.use_downsample = downsample
        
        if num_layer == 2 and downsample == 1:
            self.in_channels = 16 * (2**num_layer)
        elif num_layer > 2 and downsample != 0:
            self.in_channels = 16 * (2**(num_layer-1)) * 4
        elif downsample == 0: 
            self.in_channels = 16 * (2**num_layer) * 4
        self.out_channels = 16 * (2**num_layer)
   
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'swish':
            self.activation = nn.SiLU()
            
        if downsample == 1:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                conv1x1(self.in_channels, self.out_channels * 4, stride=1),
                nn.BatchNorm2d(self.out_channels * 4))
            self.conv1 = conv1x1(self.in_channels, self.out_channels)
            self.in_channels = self.out_channels
            self.bn1 = nn.BatchNorm2d(self.out_channels)
            self.conv2 = conv3x3(self.in_channels, self.out_channels, stride=2)
        elif downsample == -1:
            self.downsample = nn.Sequential(
                conv1x1(self.in_channels, self.out_channels * 4, stride=1),
                nn.BatchNorm2d(self.out_channels * 4))
            self.conv1 = conv1x1(self.in_channels, self.out_channels)
            self.in_channels = self.out_channels
            self.bn1 = nn.BatchNorm2d(self.out_channels)
            self.conv2 = conv3x3(self.in_channels, self.out_channels)
        elif downsample == 0:
            self.conv1 = conv1x1(self.in_channels, self.out_channels)
            self.in_channels = self.out_channels
            self.bn1 = nn.BatchNorm2d(self.out_channels)
            self.conv2 = conv3x3(self.in_channels, self.out_channels)
        
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.out_channels = 16 * (2**num_layer) * 4
        self.conv3 = conv1x1(self.in_channels, self.out_channels)
        self.bn3 = nn.BatchNorm2d(self.out_channels)
        self.in_channels = self.out_channels
        
    def forward(self, x):
        skip = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.use_downsample != 0:
            skip = self.downsample(x)

        out += skip
        out = self.activation(out)
            
        return out
