from warnings import warn
from pdb import set_trace
from multiprocessing import Process

import gym
import pygame
import torch

from torch import nn
from pygame.locals import VIDEORESIZE, RESIZABLE

class Callback(object):
    play = None
    def __init__(self): pass
    
    def __call__(self, event_name):
        """ Runs a callback method if it exists """
        res = getattr(self, event_name)()
        return res
    
    def __getattr__(self, name):
        if hasattr(self.play, name):
            return getattr(self.play, name)
    
    def __setattr__(self, name, value):
        if hasattr(self.play, name):
            msg = f"You are shadowing an attribute ({name}) that exists in the GymPlayer. " \
                  f"Use `self.play.{name}` to avoid this."
            warn(msg)
        super().__setattr__(name, None)

    def __repr__(self):
        return type(self).__name__

class KeyboardCallback(Callback): 
    def __init__(self, keys_to_action=None):
        super().__init__()
        self.keys_to_action = None
        self.pressed_keys = []
        
    def before_loop(self):
        if self.keys_to_action is None:
            if hasattr(self.env, 'get_keys_to_action'):
                self.keys_to_action = self.env.get_keys_to_action()
        elif hasattr(self.env.unwrapped, 'get_keys_to_action'):
            self.keys_to_action = self.env.unwrapped.get_keys_to_action()
        else:
            assert False, self.env.spec.id + " does not have explicit key to action mapping, " + \
                        "please specify one manually"
                        
        self.play.relevant_keys = set(sum(map(list, self.keys_to_action.keys()),[]))
    
    def set_action(self):
        self.play.action = self.keys_to_action.get(tuple(sorted(self.pressed_keys)), 0)

    def events(self):
        if self.event.type == pygame.KEYDOWN:
            if self.event.key in self.relevant_keys:
                self.pressed_keys.append(self.event.key)
        elif self.event.type == pygame.KEYUP:
            if self.event.key in self.relevant_keys:
                self.pressed_keys.remove(self.event.key)

class RandomActionCallback(Callback):
    def __init__(self):
        super().__init__()

    def set_action(self):
        self.play.action = self.env.action_space.sample()

class GymPlayer():
    def __init__(self, env, callbacks=[]):
        self.env = env
        self._callbacks = {
            'before_loop': [],
            'before_episode': [],
            'before_step': [],
            'before_action': [],
            'set_action': None,
            'after_action': [],
            'after_step': [],
            'events': [],
            'after_episode': [],
            'after_loop': [],
        }
        callbacks = [RandomActionCallback] + callbacks
        self._build_callbacks(callbacks)  

    def loop(self, n_episodes=None, transpose=True, fps=30, zoom=None, render=True):
        self.running = True
        self.render = render
        self.n_episodes = n_episodes
        self.transpose = transpose
        self.fps = fps
        self.state = None
        self.reward = None
        self.action = None
        self.next_state = None
        self.env_done = True
        self.info = None
        
        rendered = self.env.render(mode='rgb_array')
        self.video_size = [rendered.shape[1], rendered.shape[0]]
        
        if zoom is not None:
            self.video_size = int(self.video_size[0] * zoom), int(self.video_size[1] * zoom)
            
        if not render:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            
        self.screen = pygame.display.set_mode(self.video_size, RESIZABLE)
        self.clock = pygame.time.Clock()
        
        self.env.reset()
    
        self._run_event(self._do_loop, 'loop')
        
    def _do_loop(self):
        if self.n_episodes is not None:
            self._run_event(self._do_n_episodes, 'episode')
        else:
            self._run_event(self._do_episodes, 'episode')
        pygame.quit()
    
    def _do_n_episodes(self):
        for episode in range(self.n_episodes):
            self.episode = episode+1
            self.env_done = False
            self.state = self.env.reset()
            
            while not self.env_done:
                
                if not self.running: 
                    return
                
                self._run_event(self._do_step, 'step')
                
                self._do_pygame_events()
            
    def _do_episodes(self):
        self.episode = 0
        while self.running:
            
            if self.env_done:
                self.episode += 1
                self.env_done = False
                self.state = self.env.reset()
                
            self._run_event(self._do_step, 'step')
            
            self._do_pygame_events()
    
    def _do_step(self):
        self._run_event(self._do_action, 'action')

        if self.next_state is not None and self.render:
            rendered = self.env.render(mode='rgb_array')
            self.update_screen(self.screen, 
                                rendered)
    
    def _do_action(self):
        self._run_callback('set_action')
        if self.action is None:
            err = "Value action is set to None. Either no action callback is set or the " \
                    "callback does not set an action."
            raise TypeError(err)
        self.next_state, self.reward, self.env_done, self.info = self.env.step(self.action)
    
    def _do_pygame_events(self):
        for event in pygame.event.get():
            self.event = event
            self._run_callback('events')
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == VIDEORESIZE and self.render:
                self.video_size = event.size
                rendered = self.env.render(mode='rgb_array')
                self.update_screen(self.screen, 
                                    rendered)
        if self.render: 
            pygame.display.update()
            caption = f"{round(self.clock.get_fps())} {self.episode}"
            pygame.display.set_caption(caption)
        self.clock.tick(self.fps)
                    
    def _run_callback(self, cb_name):
        cbs = self._callbacks[cb_name]
        if not isinstance(cbs, list):
            cbs(cb_name)
        else:
            [cb(cb_name) for cb in cbs]
    
    def _run_event(self, event, event_name):
        self._run_callback(f'before_{event_name}')
        event()
        self._run_callback(f'after_{event_name}')
    
    def _build_callbacks(self, callbacks):
        if not isinstance(callbacks, (list, tuple)):
            callbacks = [callbacks]

        for cb in callbacks:
            if isinstance(cb, type):
                cb.play = self
                cb = cb()
            if not isinstance(cb, Callback):
                continue

            cb_dir = cb.__dir__()
            for cb_name in self._callbacks.keys():
                if cb_name in cb_dir:
                    if cb_name == 'set_action':
                        self._callbacks[cb_name] = cb
                    else:
                        self._callbacks[cb_name].append(cb)
        
    def update_screen(self, screen, arr):
        arr_min, arr_max = arr.min(), arr.max()
        arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
        pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if self.transpose else arr)
        pyg_img = pygame.transform.scale(pyg_img, self.video_size)
        screen.blit(pyg_img, (0,0))
    
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
        super(Head, self).__init__()
        self.linear_1 = nn.Linear(100,16)
        self.linear_2 = nn.Linear(16,4)
    
    def forward(self, x):
        x = x 
        x = F.relu(self.linear_1(x))

class FeedbackListener(Process):
    def __init__(self, video_size=(200, 100)):
        super().__init__()
        self.video_size = video_size
        
    def run(self, fps=30):
        self._init_pygames()
        self.listening = True
        while self.listening:
            fb, fill = self._do_pygame_events()
            self._update_screen(fill)
            self.clock.tick(fps)
            
    def _init_pygames(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.video_size, RESIZABLE)
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
            elif event.type == VIDEORESIZE:
                self.video_size = event.size
                self._update_screen(fill)
            elif event.type == pygame.QUIT:
                self.listening = False
                                 
        return fb, fill
    
    def _update_screen(self, fill=None):
        if fill is None:
            fill = self.screen.fill((0, 0, 0))
            
        pygame.display.update(fill) 
        
# TODO: Add shared memory for storing feedback
# TODO: Create Training Callbacks  
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("Bowling-v0").unwrapped 
    num_episodes = 1

    encoder = Encoder().to(device)
    head_net = Head().to(device)

    encoder.load_state_dict(torch.load("auto_encoder/Type_1/encoder.pt", map_location=device))
    
    # Freeze encoder weights
    for name, params in encoder.named_parameters():
        params.requires_grad = False

    opt = torch.optim.Adam(head_net.parameters(), lr=1e-4, weight_decay=1e-1) 
    
    listener = FeedbackListener()
    listener.start()
    
    # play = GymPlayer(env, callbacks=[KeyboardCallback])
    play = GymPlayer(env)
    play.loop(zoom=4, fps=60)
    
    listener.join()
    
if __name__ == "__main__":
    main() 
