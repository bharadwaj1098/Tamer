from pdb import set_trace

import gym
import pygame
import torch

from torch import nn
from pygame.locals import VIDEORESIZE, RESIZABLE


class Callback():
    def __init__(self):
        self.play = None
        
    def __getattr__(self, name):
        if hasattr(self.play, name):
            return getattr(self.play, name)
        
    def __repr__(self):
        return type(self).__name__

class KeyboardCallback(Callback): 
    def __init__(self, keys_to_action=None):
        super().__init__()
        self.keys_to_action = None
        self.pressed_keys = []
        
    def before_run(self):
        if self.keys_to_action is None:
            if hasattr(self.env, 'get_keys_to_action'):
                self.keys_to_action = self.env.get_keys_to_action()
        elif hasattr(self.env.unwrapped, 'get_keys_to_action'):
            self.keys_to_action = self.env.unwrapped.get_keys_to_action()
        else:
            assert False, self.env.spec.id + " does not have explicit key to action mapping, " + \
                        "please specify one manually"
                        
        self.play.relevant_keys = set(sum(map(list, self.keys_to_action.keys()),[]))
    
    def action(self):
        return self.keys_to_action.get(tuple(sorted(self.pressed_keys)), 0)
        
    def events(self):
        if self.event.type == pygame.KEYDOWN:
            if self.event.key in self.relevant_keys:
                self.pressed_keys.append(self.event.key)
        elif self.event.type == pygame.KEYUP:
            if self.event.key in self.relevant_keys:
                self.pressed_keys.remove(self.event.key)

class RandomActionCallback(Callback):
    def action(self):
        return self.env.action_space.sample()
    
class Play():
    def __init__(self, env, callbacks=[]):
        self.env = env
        self._callbacks = {
            'before_run': [],
            'action': None,
            'before_render': [],
            'after_render': [],
            'after_run': [],
            'events': []
        }
        callbacks = [RandomActionCallback] + callbacks
        self.parse_callbacks(callbacks)

    def __call__(self, n_episodes=None, transpose=True, fps=30, zoom=None,):
        """Allows one to play the game using keyboard.
        To simply play the game use:
            play(gym.make("Pong-v4"))
        Above code works also if env is wrapped, so it's particularly useful in
        verifying that the frame-level preprocessing does not render the game
        unplayable.
        If you wish to plot real time statistics as you play, you can use
        gym.utils.play.PlayPlot. Here's a sample code for plotting the reward
        for last 5 second of gameplay.
            def callback(obs_t, obs_tp1, action, rew, done, info):
                return [rew,]
            plotter = PlayPlot(callback, 30 * 5, ["reward"])
            env = gym.make("Pong-v4")
            play(env, callback=plotter.callback)
        Arguments
        ---------
        env: gym.Env
            Environment to use for playing.
        transpose: bool
            If True the output of observation is transposed.
            Defaults to true.
        fps: int
            Maximum number of steps of the environment to execute every second.
            Defaults to 30.
        zoom: float
            Make screen edge this many times bigger
        keys_to_action: dict: tuple(int) -> int or None
            Mapping from keys pressed to action performed.
            For example if pressed 'w' and space at the same time is supposed
            to trigger action number 2 then key_to_action dict would look like this:
                {
                    # ...
                    sorted(ord('w'), ord(' ')) -> 2
                    # ...
                }
            If None, default key_to_action mapping for that env is used, if provided.
        """

        self.env.reset()
        rendered = self.env.render(mode='rgb_array')
        
        video_size=[rendered.shape[1],rendered.shape[0]]
        if zoom is not None:
            video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)
            
        running = True
        env_done = True
        episode = 0

        screen = pygame.display.set_mode(video_size, RESIZABLE)
        clock = pygame.time.Clock()
        
        [cb() for cb in self._callbacks['before_run']]
        while running:
            # Stop running if episode limit reached
            if n_episodes is not None and episode > n_episodes:
                running = False
                
            # Start new episode for environment 
            if env_done:
                episode += 1
                env_done = False
                state = self.env.reset()
  
            # Get Action
            if self._callbacks.get('action') is not None:
                action = self._callbacks['action']()

            # Take Action
            prev_state = state
            state, reward, env_done, info = self.env.step(action)
            self.state_info = (prev_state, state, reward, env_done, info)

            # Render State
            [cb() for cb in self._callbacks['before_render']]
            if state is not None:
                state = self.env.render( mode='rgb_array')
                self.display_arr(screen, state, transpose=transpose, video_size=video_size)
            [cb() for cb in self._callbacks['after_render']]
            
            # Check PyGame Events
            for event in pygame.event.get():
                self.event = event
                [cb() for cb in self._callbacks['events']]
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == VIDEORESIZE:
                    video_size = event.size
                    self.display_arr(screen, rendered, transpose=transpose, video_size=video_size)

            pygame.display.flip()
            clock.tick(fps)
        [cb() for cb in self._callbacks['after_run']]
        pygame.quit()
        
    def parse_callbacks(self, callbacks):
        if not isinstance(callbacks, (list, tuple)):
            callbacks = [callbacks]

        for cb in callbacks:
            if isinstance(cb, type):
                cb = cb()
            if not isinstance(cb, Callback):
                continue
                
            cb.play = self
            for m in dir(cb):
                if m in self._callbacks:
                    if m == 'action':
                        self._callbacks[m] = getattr(cb, m)
                    else:
                        self._callbacks[m].append(getattr(cb, m))
            
        
    def display_arr(self, screen, arr, video_size, transpose):
        arr_min, arr_max = arr.min(), arr.max()
        arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
        pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
        pyg_img = pygame.transform.scale(pyg_img, video_size)
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
        x = F.rel

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

    opt = torch.optim.Adam( head_net.parameters(), lr=1e-4, weight_decay = 1e-1 ) 

    play = Play(env, callbacks=[KeyboardCallback])

    play(zoom=4, fps=60)

if __name__ == "__main__":
    main() 
