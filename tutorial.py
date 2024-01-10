import pygame
from pong import Game
import neat
import os
import pickle

width, height = 700, 500 
window = pygame.display.set_mode((width,height))

game = Game(window,width,height)


class PongGame:
    def __init__(self,window,width,height):
        self.game = Game(window,width,height)
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball 
        
    def test_ai(self,genome,config):
        # creating the neural network 
        net = neat.nn.FeedForwardNetwork.create(genome,config)
        
        run= True
        clock = pygame.time.Clock()
        while run:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False 
                    break
                
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w] :
                game.move_paddle(left=True,up=True)
            
            if keys[pygame.K_s] :
                game.move_paddle(left=True,up=False)
            
            output = net.activate((self.right_paddle.y,self.ball.y,abs(self.right_paddle.x -self.ball.x)))            
            self.ai_paddle_move(output,True)        
                
            game.loop()
            game.draw(True,False)
            pygame.display.update()
            
        pygame.quit()
        
    def train_ai(self,genome1,genome2,config):        
            net1 = neat.nn.FeedForwardNetwork.create(genome1,config)
            net2 = neat.nn.FeedForwardNetwork.create(genome2,config)
            
            run = True 
            while run:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit() 
                # The output data is 3 values 
                # We will take the max value as the suggest to the paddle movment         
                output1 = net1.activate((self.left_paddle.y,self.ball.y,abs(self.left_paddle.x -self.ball.x)))            
                self.ai_paddle_move(output1,True)
                
                output2 = net2.activate((self.right_paddle.y,self.ball.y,abs(self.right_paddle.x -self.ball.x)))            
                self.ai_paddle_move(output2,False)

                game_info = self.game.loop()        
        
                self.game.draw(draw_score=False,draw_hits=True)
                pygame.display.update()
                # If either one of the paddels missed the ball or the AI succeed 50 times we can finish         
                if game_info.left_score >=1 or game_info.right_score >=1 or game_info.left_hits > 50:
                    self.calculate_fitness(genome1,genome2,game_info)
                    break
            
    def ai_paddle_move(self,output,left):
        decision = output.index(max(output))
        # If 0 then stay still
        if decision == 0:
            pass
        # If 1 then go up 
        if decision == 1:
            self.game.move_paddle(left=left,up=True)
        # If 2 then go down 
        else:    
            self.game.move_paddle(left=left,up=False)
        
                    
    def calculate_fitness(self,genome1,genome2,game_info):
        genome1.fitness += game_info.left_hits 
        genome2.fitness += game_info.right_hits 
                
def eval_genomes(genomes,config):
    # genomes = list (genomeId,genome)
    for i,(genome_id1,genome1) in enumerate(genomes):        
        if i == len(genomes) -1:
            break
        genome1.fitness = 0
        for genome_id2, genome2 in genomes[i+1:]:
            # Init the fitness in the first time 
            genome2.fitness = 0 if genome2.fitness == None else genome2.fitness     
            game = PongGame(window,width,height)
            game.train_ai(genome1,genome2,config)
        
def run_neat(config):
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-8')
    # p = neat.Population(config)
    # Output to console the data of generation,best fitness and etc 
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # Create a check point of the current status of the neural network 
    p.add_reporter(neat.Checkpointer(1))
    
    winner = p.run(eval_genomes,50)
    with open("best.pickle","wb") as f:      
        pickle.dump(winner,f)

def test_ai(config):
    with open("best.pickle","wb") as f:
        winner = pickle.load(f)
    game = PongGame(window,width,height)
    game.test_ai(winner,config)
    
    
if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir,"config.txt")
    
    config = neat.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,config_path)

    # run_neat(config)
    test_ai(config)
