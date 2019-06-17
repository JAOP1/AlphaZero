
GAMEROWS2 = 6
GAMECOLS2 = 7


class BoardGame1:
    
    def __init__(self,rows,cols):
        self.rows = rows
        self.cols = cols
        
        self.BoardState = [[0 for col in range(cols)] for row in range(rows)]
        self.stones_stack = [-1 for i in range(cols)]
        
        
        
    def available_actions(self,player):
        actions = []
        
        for i in range(len(self.stones_stack)):
            if self.stones_stack[i] + 1 < self.rows:
                actions.append([self.stones_stack[i] +1 , i, player])
                
        return actions
    
    
    def update_boardgame(self, action,valor):
        X,Y = action[0], action[1]
        player = action[2]
        self.BoardState[X][Y] = player
        self.stones_stack[Y] += valor


class conecta4:

    def children(self,boardgame,player):
        actions = boardgame.available_actions(player)
    
        return actions


    def is_complete(self,boardgame,last_action_taken):

        def is_winner_move(x,y,stepX,stepY,player,boardgame):
            counter = 0
            boardgame1 = boardgame.BoardState
            while -1 < x < GAMEROWS2 and -1 < y < GAMECOLS2 and counter<4:
                if  boardgame1[x][y] != player:
                    return False

                counter += 1
                x += stepX
                y += stepY

            return (True if counter == 4 else False)
        #Diagonal Derecha superior, Derecha, Diagonal derecha inferior, Diagonal izquierda superior
        #Diagonal izquierda inferior, para bajo , izquierda.
        directions = [[-1,1],[0,1],[0,-1],[-1,0],[-1,-1]]
        X,Y = last_action_taken[0],last_action_taken[1]
        player = last_action_taken[2]

        for direction in directions:
            if is_winner_move(X,Y,direction[0],direction[1],player,boardgame):
                return True

        return False


    
    def reward(self,BoardGame,last_action):
      
        
        if self.is_complete(BoardGame,last_action):
            return 1
        
        return 0

    def execute_action(self,boardgame,action):
        boardgame.update_boardgame(action,1)
        


    def undo_action(self,boardgame,action):
        action[2] = 0
        boardgame.update_boardgame(action,-1)

