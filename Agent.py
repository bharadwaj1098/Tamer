from warnings import warn
from pdb import set_trace
from multiprocessing import Process, Queue
from functools import partial
from play import PyGymCallback, Player, FeedbackListener

import gym
import pygame
import os 
import numpy as np
import time 
import datetime as dt 

import torch
import torch.nn.functional as F
import torchvision.transforms as T 
import torch.optim as optim
from torch import nn

torch.manual_seed(10)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
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

class Head(nn.Module):
    def __init__(self):
        super(Head,self).__init__()

        self.linear_1 = nn.Linear(100,16)
        self.linear_2 = nn.Linear(16,4)
    
    def forward(self, x):
        x = x 
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        return x

class NetworkController(PyGymCallback):
    def __init__(self, encoder, head, queue, img_dims = (3, 160, 160), ts_len = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.head = head
        self.queue = queue 
        self.img_dims = img_dims
        self.ts_len = ts_len
        self.buffer = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def before_set_action(self):
        state = self.env.render(mode='rgb_array').transpose((2,0,1))
        state = np.ascontiguousarray(state, dtype = np.float32)/255
        state = torch.from_numpy(state)
        resize = T.Compose([T.ToPILImage(),
                            T.Resize((self.img_dims[1:])),
                            T.ToTensor()])
        self.state_ts = time.time()
        state = resize(state).to(self.device).unsqueeze(0) 
        return state  
    
    def set_action(self):
        self.play.state = self.before_set_action()
        self.network_output = self.head(self.encoder(self.play.state.to(self.device)))
        self.action = np.argmax(self.network_output.detach().numpy())
        self.play.action = self.action

        fb = self.queue.get()
        if  fb != 0:
            self.buffer.append([self.play.state, fb, np.amax(self.network_output.detach().numpy())])

    def after_set_action(self):
        '''
        should sample over buffer and perform SGD\
            but, how to update the weights of Head_network ?
        '''
        if len(self.buffer) > 10:
            print('working')
            for i in self.buffer:
                kk = f"state_shape : {i[0].shape} feedback : {i[1]} network_output : {i[2]}"
                print(kk)

class FeedbackListener(Process):
    def __init__(self,fb_queue,video_size=(200, 100)):
        super().__init__()
        self.video_size = video_size
        self.fb_queue = fb_queue
        
    def run(self, fps=30):
        self._init_pygames()
        self.listening = True
        while self.listening:
            fb, fill = self._do_pygame_events()
            self._update_screen(fill)
            #add feedback to queue is feeback =! 0
            self.clock.tick(fps)
            self.fb_queue.put(fb)

    def _init_pygames(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.video_size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self._update_screen()
    
    def _do_pygame_events(self):
        fb, fill = 0, None
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    fill = self.screen.fill((0, 255, 0))
                    fb = 1
                elif event.key == pygame.K_2:
                    fill = self.screen.fill((255, 0, 0))
                    fb = -1
            elif event.type == pygame.VIDEORESIZE:
                self.video_size = event.size
                self._update_screen(fill)
            elif event.type == pygame.QUIT:
                self.listening = False
        return fb, fill 
    
    def _update_screen(self, fill=None):
        if fill is None:
            fill = self.screen.fill((0, 0, 0))
            
        pygame.display.update(fill) 

def main():
    env = gym.make("Bowling-v0").unwrapped 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_episodes = 1

    encoder = Encoder().to(device)
    head_net = Head().to(device) 
    encoder.load_state_dict(torch.load("auto_encoder/Type_1/encoder.pt", map_location=device))
    
    # Freeze encoder weights
    for name, params in encoder.named_parameters():
        params.requires_grad = False

    opt = torch.optim.Adam(head_net.parameters(), lr=1e-4, weight_decay=1e-1) 
    
    Feedback_queue = Queue()

    listener = FeedbackListener(Feedback_queue) #pass it to listener ()
    listener.start()
    player = Player(callbacks=[NetworkController(encoder= encoder, head=head_net, queue=Feedback_queue,
                                                    env=env, zoom=4, fps=60, human=True)]) #pass the queue
    player.play()
    listener.join()
    

if __name__ == "__main__":
    main() 