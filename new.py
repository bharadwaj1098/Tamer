import pygame
pygame.init()
WIDTH, HEIGHT = 600, 250
WIN = pygame.display.set_mode((WIDTH, HEIGHT)) 

def main():
    run = True
    while run:
        events = pygame.event.get()
        if len(events) > 0:
            print(events)
        for event in events:
            print(event.type)
            if event.type == pygame.QUIT:
                run = False 
    pygame.quit() 

class Test():
    def name(self):
        print('test')
class Test2(Test):
    def name(self):
        super().name()
        print('test2')
if __name__ == '__main__':
    test = Test2()
    test.name()