from Utilities.GameRules import *
from Utilities.GameRules1 import *
import os
from pathlib import Path
#Location
GAMEROWS = 6
GAMECOLS = 7


current_path = Path(os.getcwd())

PATHMODELS = str(current_path) + '/Models'
print(PATHMODELS)
#Game parameters.
NAMEGAME = "Gomoku"

NULLACTION = [-1,-1,-1]

def Initial_state():
    return BoardGame1(GAMEROWS,GAMECOLS)

#Monte Carlo Search parameters.
C_MCTS = 2.0
EXECUTE_MCTS = 100
GameRules = conecta4()



#AlphaZero parameters
THRESHOLD = 0.60
MAXSIZEBUFFER = 1500
BATCH_SIZE = 128
TRAIN_ROUND = 7
COMPETITION_TIME = 2
ROUNDS = 100
LEARNING_RATE = 0.01
OUTPUTS = 7 

OBS_SHAPE = (2, GAMEROWS, GAMECOLS) 


#Create action directory 
def vector_to_index(action):
    row = action[0]
    col = action[1]
    return  col 

def index_to_vector(index):
    row = index // GAMECOLS
    col = index % GAMECOLS 
    return [row,col]
    
