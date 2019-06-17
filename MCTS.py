"""
MCTS for AlphaZero
"""

import numpy as np
import copy
from Utilities.Parameters import *
from Utilities.transform import *




class Node:
    def __init__(self,CurrentAction = None,prior = None, parent = None):
        self.Parent = parent
        self.Action = CurrentAction
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.Children = []

    def add_child(self, action, prior):
        self.Children.append(Node(action,prior,self))

    def update_information(self, value):
        self.visit_count += 1
        self.total_value += value

    def confidence_node(self):

        N = (0 if self.Parent == None else  self.Parent.visit_count)
        Q = 0.0

        if self.visit_count != 0:
            Q = self.total_value / self.visit_count

        p = self.prior
        n = self.visit_count

        return Q + Parameters.C_MCTS * p * (np.sqrt(N) / (n+1))

    def is_leaf(self):
        if len(self.Children):
            return False
        return True



class MonteCarloTree:

    def __init__(self):

        self.TimesToRepeat = EXECUTE_MCTS
        self.GameRules = GameRules
        self.GameState = None
        self.net  = None


    def search(self, GameState, net):

        self.net = net
        self.GameState = copy.deepcopy(GameState)
        self.root = Node()

        for iteration in range(self.TimesToRepeat):

            SelectedNode, BoardGame , deep = self.select(self.root)
           

            children, values = self.expand(SelectedNode , BoardGame,deep)
          
            for i in range(len(values)):

                self.backpropagation(children[i],values[i])

    def clear(self):
        self.net = None
        self.root = None

    def Get_probabilities(self):
        
        probabilities = np.zeros(OUTPUTS)

        for child in self.root.Children:
            index = vector_to_index(child.Action)
            probabilities[index] = child.visit_count
      

        total = np.sum(probabilities)

        probabilities = probabilities / total

        return probabilities

    def backpropagation(self, leaf, value):
        CurrentNode = leaf

        while CurrentNode.Parent is not None:
            CurrentNode.update_information(value)
            CurrentNode = CurrentNode.Parent
            value *= -1


    def expand(self, node, GameState, deep):
        BoardGame = copy.deepcopy(GameState)

        type_reward = (1 if deep % 2 != 0 else -1)
        player = (1 if deep % 2 != 0 else 2)
        values_net = []

        actions = self.GameRules.children(GameState,player)
      
        for NewAction in actions:
            self.GameRules.execute_action(BoardGame,NewAction)
            batch = state_lists_to_batch([BoardGame], [player])
            priors,value = self.net(batch)
            
            values_net.append(type_reward * value)
            
       
            ind = vector_to_index(NewAction)
            node.add_child(NewAction,priors[0][ind])

            self.GameRules.undo_action(BoardGame,NewAction)

        return node.Children, values_net


    def select(self, node):
        selected_node = node
        board_game = copy.deepcopy(self.GameState)
        deep_num = 1

        def child_highest_confidence(node):

            better_child = node.Children[0]

            for child in node.Children:

                if child.confidence_node() > better_child.confidence_node():
                    better_child = child

            return better_child


        while not selected_node.is_leaf():
            selected_node = child_highest_confidence(selected_node)
            self.GameRules.execute_action(board_game, selected_node.Action)
            deep_num += 1

        return selected_node,board_game,deep_num

