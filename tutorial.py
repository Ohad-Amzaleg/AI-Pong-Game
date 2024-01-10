import pygame
from pong import Game
import neat
import os
import pickle

width, height = 700, 500 
window = pygame.display.set_mode((width,height))

game = Game(window,width,height)


class PongGame:
    def __init__(self,windows,width,height):
        self.game = Game(window,width,height)
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball 
        
    def test_ai(self):
        run= True
        clock = pygame.time.Clock()
        while run:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False 
                    break
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                game.move_paddle(left=True,up=True)
                
            if keys[pygame.K_s]:
                game.move_paddle(left=True,up=False)
                
                
            game.loop()
            game.draw(False,True)
            pygame.display.update()
        pygame.quit()

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir,"config.txt")
    
    config = neat.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,config_path)
