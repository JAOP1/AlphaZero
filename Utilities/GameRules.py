"""
Gomoku
"""

GAMECOLS1 = 7
GAMEROWS1 = 7





class gomoku:

    def children(self,boardgame,player):
        actions = []
        for row in range(GAMEROWS1):
            for col in range(GAMECOLS1):
                if boardgame[row][col] == 0:
                    actions.append([row,col,player])

        return actions


    def is_complete(self,boardgame,last_action_taken):

        def is_winner_move(x,y,stepX,stepY,player,boardgame):
            counter = 0
            while -1 < x < GAMEROWS1 and -1 < y < GAMECOLS1 and counter<5:
                if  boardgame[x][y] != player:
                    return False

                counter += 1
                x += stepX
                y += stepY

            return (True if counter == 5 else False)
        #Diagonal Derecha superior, Derecha, Diagonal derecha inferior, Diagonal izquierda superior
        #Diagonal izquierda inferior, para bajo , izquierda.
        directions = [[-1,1],[0,1],[1,1],[-1,-1],[1,-1],[1,0],[-1,0],[0,-1]]
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
        row = action[0]
        col = action[1]
        player = action[2]
        boardgame[row][col] = player


    def undo_action(self,boardgame,action):
        row = action[0]
        col = action[1]
        boardgame[row][col] = 0

