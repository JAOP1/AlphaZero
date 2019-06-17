import Utilities.Parameters as Parameters
import Model
from MCTS import MonteCarloTree
from Utilities.transform import state_lists_to_batch
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import collections
import random
import os


def Update_reward_game(game,reward):

    for i in range(len(game)):
        game[-(i+1)][-1] = reward
        reward *= -1


# Return all the game played and the reward of the first player
def combat_game(Agent1, Agent2, save_history=True):
    players = [Agent1, Agent2]
    BufferGame = []
    Board = Parameters.Initial_state()
    current_action = Parameters.NULLACTION
    current_player = 0
    frames = 0
    
    while not Parameters.GameRules.is_complete(Board, current_action) and frames < Parameters.GAMEROWS*Parameters.GAMECOLS:
        current_action, probs = players[current_player].take_action(Board)
     
        current_action[2] = current_player+1
        
        Parameters.GameRules.execute_action(Board, current_action)
        
        if save_history:
            current_board = copy.deepcopy(Board)
          
            BufferGame.append(
                [current_board, current_player+1, probs, None])  # Game state, player, probabilities of actions and reward

        current_player = 1 - current_player
        frames += 1
        
    print("Acabo una fase de juego.")
    reward = Parameters.GameRules.reward(Board,current_action)
    Update_reward_game(BufferGame,reward)

    reward *= (-1 if frames%2 == 0 else 1)

    return BufferGame, reward

# Portion where the first agent win to the second one.
def evaluate(Agent1, Agent2):
    Agent1_won = 0
    Agent2_won = 0
    
    for round in range(Parameters.ROUNDS):
        game, reward = combat_game(Agent1, Agent2, save_history=False)

        if reward > 0.4:
            Agent1_won += 1
        
        elif reward <0.4:
            Agent2_won +=1
            
    return Agent1_won / (Agent1_won + Agent2_won)




def load_net(Path_model,Agent):
    Agent.net.load_state_dict(torch.load(Path_model))
    Agent.net.train()


class AgentZero:

    def __init__(self,device):
        self.net = Model.Net(input_shape=Parameters.OBS_SHAPE, actions_n=Parameters.OUTPUTS).to(device)
        self.MCT = MonteCarloTree()
        self.optimizer = optim.SGD(self.net.parameters(), lr=Parameters.LEARNING_RATE, momentum=0.9)
        
        self.sum_loss = 0.0
        self.sum_value_loss=0.0
        self.sum_policy_loss=0.0
        
    def take_action(self,BoarGame):
        
        self.MCT.search(BoarGame,self.net)

        probs = self.MCT.Get_probabilities()
        choose = np.random.choice(len(probs) , 1 , p=probs)[0]
        #action =Parameters.index_to_vector(choose) + [0]
       
        action = [BoarGame.stones_stack[choose] + 1,choose,0]
        return action, probs
    
    def clear_stats(self):
        self.sum_loss = 0.0
        self.sum_value_loss = 0.0
        self.sum_policy_loss = 0.0
    
    def show_stats(self):
        print("Total loss: %f" % (self.sum_loss/Parameters.TRAIN_ROUND))
        print("Value loss: %f" % (self.sum_value_loss/Parameters.TRAIN_ROUND))
        print("Policy loss: %f" % (self.sum_policy_loss/Parameters.TRAIN_ROUND))
        
        self.clear_stats()

    def fit(self,batch_probs,batch_values,states_v,device = 'cuda'):
        self.optimizer.zero_grad()
        probs_v = torch.FloatTensor(batch_probs).to(device)
        values_v = torch.FloatTensor(batch_values).to(device)
        out_logits_v, out_values_v = self.net(states_v)

        loss_value_v = F.mse_loss(out_values_v.squeeze(-1), values_v)
        loss_policy_v = -F.log_softmax(out_logits_v, dim=1) * probs_v
        loss_policy_v = loss_policy_v.sum(dim=1).mean()

        loss_v = loss_policy_v + loss_value_v
        loss_v.backward()
        self.optimizer.step()

        self.sum_loss += loss_v.item()
        self.sum_value_loss += loss_value_v.item()
        self.sum_policy_loss += loss_policy_v.item()




class AlphaZero:

    def __init__(self,stop_condition):
        self.Master = AgentZero('cuda')
        self.student = AgentZero('cuda')
        
        
        self.examples = collections.deque(maxlen=Parameters.MAXSIZEBUFFER)
        self.stop_condition = stop_condition
        self.steps = 1
        self.switch = 0

    def create_dataset(self):
        steps = 0
        while len(self.examples) < Parameters.MAXSIZEBUFFER:
            game_steps, reward_first_player = combat_game(self.Master,self.Master)
            self.examples.extend(game_steps)
            print(len(self.examples))
          

    def data_selection(self):
        batch = random.sample(self.examples, Parameters.BATCH_SIZE)  
    
        batch_states,batch_players,batch_probabilities,batch_reward = zip(*batch)
      
        #print(batch_states)
        batch_states = state_lists_to_batch(batch_states,batch_players)
        
        return [batch_states,batch_players,batch_probabilities,batch_reward]


    def training_phase(self):

        while self.stop_condition(self.steps):

            self.create_dataset()
            print("------------------------")
            
            for iteration  in range(Parameters.TRAIN_ROUND):
        
                data = self.data_selection()
              
                #print("Conjunto de datos seleccionado")

                self.student.fit(data[2],data[3], data[0])

            self.student.show_stats()
            
            if self.steps % Parameters.COMPETITION_TIME == 0:


                if evaluate(self.student,self.Master) > Parameters.THRESHOLD:
                    file = Parameters.PATHMODELS + '/Model_%d_%d.pth'%(self.steps,self.switch)
                    
                    torch.save(self.student.net.state_dict(), file)
                    
                    load_net(file,self.Master)
                    
                    print("Found best player")
              
            self.switch += 1
            self.steps +=1
            self.examples.clear()
            
        self.steps = 1
