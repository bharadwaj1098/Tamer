import torch as T 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

class encoder(nn.Module):
    '''
    def __init__(self):
        super(encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 1, 3)

        self.conv_bn1 = nn.BatchNorm2d(64)
        self.conv_bn2 = nn.BatchNorm2d(1)
        '''
    def __init__(self):
        super(encoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3),
            nn.BatchNorm2d(1),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
    def forward(self, x):

        x = F.max_pool2d( self.conv_bn1(self.conv1(x)), 2)
        x = F.relu(x)

        x = F.max_pool2d( self.conv_bn1(self.conv2(x)), 2)
        x = F.relu(x)

        x = F.max_pool2d( self.conv_bn1(self.conv2(x)), 2)
        x = F.relu(x) 

        x = F.max_pool2d( self.conv_bn2(self.conv3(x)), 2)
        x = F.relu(x) 
        
        return x

encoder = encoder()

print(encoder)
'''
for i in encoder.parameters():
    print(i.shape)
'''