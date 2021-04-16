from torch import nn

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

def conv3x3(in_channels, out_channels, stride=1,padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)

class ResNet_Normal_Block(nn.Module):
    def __init__(
        self,
        num_layer,
        downsample = 0,
        ):
        super(ResNet_Normal_Block, self).__init__()
        self.use_downsample = downsample
        if num_layer == 2 and downsample == 1:
            self.in_channels = 16*(2**num_layer)
        elif num_layer > 2 and downsample != 0:
            self.in_channels = 16*(2**(num_layer-1))
        elif downsample == 0: 
            self.in_channels = 16*(2**num_layer)
            
        self.out_channels = 16*(2**num_layer)
        
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
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(self.in_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        
    def forward(self, x):
        skip = x
#         print('Block input',x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
#         print('Block before skip',out.shape)

        if self.use_downsample != 0:
#             print('Before downsample',out.shape, skip.shape)
            skip = self.downsample(x)
#             print('After downsample',out.shape, skip.shape)
        out += skip
        out = self.relu(out)

        return out
    
class ResNet_Bottleneck_Block(nn.Module):
    def __init__(
        self,
        num_layer,
        downsample = 0
        
    ):
        super(ResNet_Bottleneck_Block, self).__init__()
        
        self.use_downsample = downsample
        if num_layer == 2 and downsample == 1:
            self.in_channels = 16*(2**num_layer)
        elif num_layer > 2 and downsample != 0:
            self.in_channels = 16*(2**(num_layer-1))*4
        elif downsample == 0: 
            self.in_channels = 16*(2**num_layer)*4
            
        self.out_channels = 16*(2**num_layer)
        
        if downsample == 1:
            self.downsample = nn.Sequential(
                conv1x1(self.in_channels, self.out_channels*4, stride=2),
                nn.BatchNorm2d(self.out_channels*4))
            self.conv1 = conv1x1(self.in_channels, self.out_channels,stride=2)
            self.in_channels = self.out_channels
            self.bn1 = nn.BatchNorm2d(self.out_channels)
        elif downsample == -1:
            self.downsample = nn.Sequential(
                conv1x1(self.in_channels, self.out_channels*4, stride=1),
                nn.BatchNorm2d(self.out_channels*4))
            self.conv1 = conv1x1(self.in_channels, self.out_channels)
            self.in_channels = self.out_channels
            self.bn1 = nn.BatchNorm2d(self.out_channels)
        elif downsample == 0:
            self.conv1 = conv1x1(self.in_channels, self.out_channels)
            self.in_channels = self.out_channels
            self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(self.in_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU()
        self.out_channels = 16*(2**num_layer)*4
        self.conv3 = conv1x1(self.in_channels, self.out_channels)
        self.bn3 = nn.BatchNorm2d(self.out_channels)
        self.in_channels = self.out_channels
        
    def forward(self, x):
        skip = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.use_downsample != 0:
#             print(out.shape, x.shape)
            skip = self.downsample(x)
#             print(out.shape, x.shape)
        out += skip
        out = self.relu(out)
            
        return out
    
class ResNet_D_Normal_Block(nn.Module):
    def __init__(
        self,
        num_layer,
        downsample = 0,
    ):
        super(ResNet_D_Normal_Block, self).__init__()
        self.use_downsample = downsample
        if num_layer == 2 and downsample == 1:
            self.in_channels = 16*(2**num_layer)
        if num_layer > 2 and downsample != 0:
            self.in_channels = 16*(2**(num_layer-1))
        elif downsample == 0: 
            self.in_channels = 16*(2**num_layer)
            
        self.out_channels = 16*(2**num_layer)
        
        if downsample == 1:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2,stride=2),
                conv1x1(self.in_channels, self.out_channels, stride=1),
                nn.BatchNorm2d(self.out_channels))

            self.conv1 = conv3x3(self.in_channels, self.out_channels,stride=2)
        elif downsample == -1:
            self.downsample = conv1x1(self.in_channels, self.out_channels, stride=1)
            
            self.conv1 = conv3x3(self.in_channels, self.out_channels, stride=1)
            
        elif downsample == 0:
            self.conv1 = conv3x3(self.in_channels, self.out_channels, stride=1)
        self.in_channels = self.out_channels
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(self.in_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        
    def forward(self, x):
        skip = x
#         print('Block input',x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
#         print('Block before skip',out.shape)

        if self.use_downsample != 0:
#             print('Before downsample',out.shape, skip.shape)
            skip = self.downsample(x)
#             print('After downsample',out.shape, skip.shape)
        out += skip
        out = self.relu(out)

        return out
    
class ResNet_D_Bottleneck_Block(nn.Module):
    def __init__(
        self,
        num_layer,
        downsample = 0
        
    ):
        super(ResNet_D_Bottleneck_Block, self).__init__()
        
        self.use_downsample = downsample
        if num_layer == 2 and downsample == 1:
            self.in_channels = 16*(2**num_layer)
        elif num_layer > 2 and downsample != 0:
            self.in_channels = 16*(2**(num_layer-1))*4
        elif downsample == 0: 
            self.in_channels = 16*(2**num_layer)*4
            
        self.out_channels = 16*(2**num_layer)
   
        if downsample == 1:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2,stride=2),
                conv1x1(self.in_channels, self.out_channels*4, stride=1),
                nn.BatchNorm2d(self.out_channels*4))

            self.conv1 = conv1x1(self.in_channels, self.out_channels)
            self.in_channels = self.out_channels
            self.bn1 = nn.BatchNorm2d(self.out_channels)
            self.conv2 = conv3x3(self.in_channels, self.out_channels,stride=2)
        elif downsample == -1:
            self.downsample = nn.Sequential(
#                 nn.AvgPool2d(kernel_size=2,stride=2)
                conv1x1(self.in_channels, self.out_channels*4, stride=1),
                nn.BatchNorm2d(self.out_channels*4))
            
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
        self.out_channels = 16*(2**num_layer)*4
        self.conv3 = conv1x1(self.in_channels, self.out_channels)
        self.bn3 = nn.BatchNorm2d(self.out_channels)
        self.in_channels = self.out_channels
        self.relu = nn.ReLU()
        
    def forward(self, x):
        skip = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.use_downsample != 0:
#             print(out.shape, x.shape)
#             if self.use_downsample == 1:
#                 out = self.maxpool(out)
            skip = self.downsample(x)
#             print(out.shape, x.shape)
        out += skip
        out = self.relu(out)
            
        return out