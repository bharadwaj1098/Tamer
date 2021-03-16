import gym
import numpy as np 
import time
import random
import cv2
import matplotlib.pyplot as plt
from collections import namedtuple
import warnings 
warnings.filterwarnings("ignore") 

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.transforms as T 
import torch.optim as optim 

env = gym.make("Bowling-v0").unwrapped 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_actions = env.action_space.n
no_of_episodes = 10000
no_of_steps = 200
batch_size = 128        
buffer_capacity = 150

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 1, 3)

        self.conv_bn1 = nn.BatchNorm2d(64)
        self.conv_bn2 = nn.BatchNorm2d(1)
        
        self.linear_1 = nn.Linear(64, 100)
        
    def forward(self, x):
        x = x
        x = F.max_pool2d(self.conv_bn1(self.conv1(x)), 2)
        x = F.max_pool2d(self.conv_bn1(self.conv2(x)), 2)
        x = F.max_pool2d(self.conv_bn1(self.conv2(x)), 2)
        x = F.max_pool2d(self.conv_bn2(self.conv3(x)), 2)
        x = x.view(x.size(0), -1)
        x = self.linear_1(x) #encoded states might come in "-ve" so no Relu or softmax
        return x

class decoder(nn.Module):
    #Linear - Reshape - Upsample - Deconv + BatchNorm 
    def __init__(self):
        super(decoder, self).__init__()

        self.linear_1 = nn.Linear(100,64)

        self.upsample_1 = nn.Upsample(scale_factor=2)
        self.upsample_2 = nn.Upsample(scale_factor=2.03)

        self.deconv_1 = nn.ConvTranspose2d(1, 64, 3)
        self.deconv_2 = nn.ConvTranspose2d(64, 64, 3)
        self.deconv_3 = nn.ConvTranspose2d(64, 3, 3)

        self.deconv_bn1 = nn.BatchNorm2d(64)
        self.deconv_bn2 = nn.BatchNorm2d(3)  

    def forward(self,x):
        x = x
        x = F.relu(self.linear_1(x))  
        x = x.view(x.shape[0],1,8,8)
        x = self.deconv_bn1(self.deconv_1(self.upsample_1(x)))
        x = self.deconv_bn1( self.deconv_2(self.upsample_1(x)))
        x = self.deconv_bn1(self.deconv_2(self.upsample_2(x)))
        x = self.deconv_bn2(self.deconv_3(self.upsample_1(x))) 
        return x
'''
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 1, 3)

        self.conv_bn1 = nn.BatchNorm2d(64)
        self.conv_bn2 = nn.BatchNorm2d(1)

        self.linear_1 = nn.Linear(64, 100)
        self.linear_2 = nn.Linear(100,16)
        self.linear_3 = nn.Linear(16,4)

    def forward(self, x):
        x = x
        x = F.max_pool2d(self.conv_bn1(self.conv1(x)), 2) 
        x = F.max_pool2d(self.conv_bn1(self.conv2(x)), 2) 
        x = F.max_pool2d(self.conv_bn1(self.conv2(x)), 2) 
        x = F.max_pool2d(self.conv_bn2(self.conv3(x)), 2) 
        x = x.view(x.size(0), -1)
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        return x
'''

Results = namedtuple(
    'ORIGINAL',
    ('original') 
)

class Buffer():
    def __init__(self, capacity):
        self.capacity = capacity 
        self.memory = []
        self.push_count = 0
    
    def push(self, Results):
        if len(self.memory)< self.capacity:
            self.memory.append(Results)
        else:
            self.memory[self.push_count % self.capacity] = Results 
        self.push_count += 1

    def random_sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size 
    
    def empty(self):
        self.memory.clear()

def select_action():
    action = torch.tensor([[random.randrange(n_actions)]], device = device, dtype = torch.long)
    return action

def get_screen():
    screen = env.render(mode = "rgb_array").transpose((2,0,1))
    screen = np.ascontiguousarray(screen, dtype = np.float32)/255
    screen = torch.from_numpy(screen)
    resize = T.Compose([T.ToPILImage(),
                        T.Resize((160, 160)),
                        T.ToTensor()])
    screen = resize(screen).unsqueeze(0).to(device)
    return screen 

class custom_loss(nn.Module):
    def __init__(self):
        super(custom_loss,self).__init__()
          
        self.sum_of_all = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def change_to_image(self, img):
        img = img.permute(1, 2, 0).cpu() 
        return img

    def MSE(self, img1, img2):
        img1 = img1.data.numpy()
        img2 = img2.data.numpy()
        err = np.sum((img1.astype('float') - img2.astype('float')) ** 2)
        err /= float(img1.shape[0] * img1.shape[1])
        return err
      
    def Forward(self, original, output):
        M = original.shape[0] 
        for i in range(M): 
            x = self.change_to_image(original[i])
            y = self.change_to_image(output[i])
            error = self.MSE(x, y)
            self.sum_of_all = self.sum_of_all + error
        loss = self.sum_of_all / M
        loss = torch.tensor([loss], device = self.device, requires_grad = True)
        return loss 
     
if __name__ == "__main__":

    buffer = Buffer(buffer_capacity)
    loss = custom_loss()
    loss_history = []

    encoder = encoder()
    decoder = decoder()

    encoder.to(device)
    decoder.to(device)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=1e-3)

    for episode in range(no_of_episodes):
        env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen
        encoder_optimizer.zero_grad() 
        decoder_optimizer.zero_grad() 
        Done = False
        for steps in range(no_of_steps):
            action = select_action() 
            _, _, Done, _ = env.step(action.item())
            last_screen = current_screen
            current_screen = get_screen()
            next_state = current_screen - last_screen
            if Done:
                next_state = None 
            buffer.push((state))
            state = next_state
        if buffer.can_sample(batch_size): 
            state_batch = buffer.random_sample(batch_size)
        else:
            state_batch = buffer.random_sample(len(buffer.memory)) 
        
        state_batch = torch.stack(state_batch).float() # to convert list to a shape of [batch_size, 1, 3, 160, 160]
        state_batch = np.squeeze(state_batch, axis = 1) 
        output_batch = decoder(encoder(state_batch.cuda())) 

        current_loss = loss.Forward(state_batch, output_batch)
        loss_history.append(current_loss.cpu())
        if episode % 500 == 0:
            print("Episode_number = {}\n Current_Loss = {}\n ".format(episode, 
                                                                    current_loss.item()))
        current_loss.backward() 
        decoder_optimizer.step()
        encoder_optimizer.step()
        buffer.empty()

    torch.save(encoder.state_dict(), "encoder")
    torch.save(decoder.state_dict(), "decoder")




