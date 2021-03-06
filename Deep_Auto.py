import torch as T 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 1, 3)

        self.conv_bn1 = nn.BatchNorm2d(64)
        self.conv_bn2 = nn.BatchNorm2d(1)

        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        def max2d_size_out(size, kernel_size = 2, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1 

        linear_input_size = max2d_size_out(conv2d_size_out(max2d_size_out( conv2d_size_out(160))))
        linear_input_size = max2d_size_out(conv2d_size_out(max2d_size_out( conv2d_size_out(linear_input_size))))

        self.linear_1 = nn.Linear(linear_input_size^2, 100)

    def forward(self, x):
        x = x
        x = F.max_pool2d( self.conv_bn1(self.conv1(x)), 2)
        x = F.relu(x)

        x = F.max_pool2d( self.conv_bn1(self.conv2(x)), 2)
        x = F.relu(x)

        x = F.max_pool2d( self.conv_bn1(self.conv2(x)), 2)
        x = F.relu(x) 

        x = F.max_pool2d( self.conv_bn2(self.conv3(x)), 2)
        x = F.relu(x) 
        
        x = F.relu(self.linear_1(x.view(x.size(0), -1)))

        return x


class decoder(nn.module):
    
    def __init__(self):
        super(decoder, self).__init()
