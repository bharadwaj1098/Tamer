import pygame
pygame.init()
WIDTH, HEIGHT = 600, 250
WIN = pygame.display.set_mode((WIDTH, HEIGHT)) 

def main():
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False 
    pygame.quit() 

if __name__ == '__main__':
    main() 