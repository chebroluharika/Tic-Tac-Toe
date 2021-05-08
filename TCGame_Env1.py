from gym import spaces
import numpy as np
import random
from itertools import groupby
from itertools import product



class TicTacToe:
    def __init__(self):
        # The 3*3 board is represented as an array of 9 values
        # 
        # 0 1 2
        # 3 4 5
        # 6 7 8
        self.board = [0]*9
        
        # initialise state as an array
        self.state = [np.nan for _ in range(9)]  # initialises the board position, can initialise to an array
        # all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)] # , can initialise to an array
        
        # player_1 i.e., Odd player has only options [1, 3, 5, 7, 9]
        self.player_1 = None
        # player_2 i.e., Even player has only options [2, 4, 6, 8]
        self.player_2 = None

    #reset the game
    def reset(self):
        self.board = [0] * 9

   #evaluate function
    def is_winning(self):
        """Takes state as an input and returns whether any row, column or diagonal has winning sum
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False"""
        # "row checking"
        for j in range(3):
            if (self.board[j * 3] + self.board[j * 3 + 1] + self.board[j * 3 + 2]) == 15:
                return 1.0, True
        
        # "col checking"
        for k in range(0,3):
            if (self.board[k + 0] + self.board[k + 3] + self.board[k + 6]) == 15:
                return 1.0, True
        
        # diagonal checking
        if (self.board[0] + self.board[4] + self.board[8]) == 15:
            return 1.0, True
        if (self.board[2] + self.board[4] + self.board[6]) == 15:
            return 1.0, True

        if not any(blank == 0 for blank in self.board):
            return 0.0, True

        return 0.0, False
    

    def is_terminal(self, curr_state):
        # Terminal state could be winning state or when the board is filled up

        if self.is_winning(curr_state) == True:
            return True, 'Win'

        elif len(self.allowed_positions(curr_state)) ==0:
            return True, 'Tie'

        else:
            return False, 'Resume'    

    #return remaining possible moves
    def allowed_moves(self):
        possible_moves = [blanks + 1 for blanks, spot in enumerate(self.board) if spot == 0]
        return possible_moves

    def allowed_values(self, curr_state):
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board"""

        player_1.options = [val for val in curr_state if not np.isnan(val)]
        player_1.options = [val for val in self.all_possible_numbers if val not in used_values and val % 2 !=0]
        player_2.options = [val for val in self.all_possible_numbers if val not in used_values and val % 2 ==0]

        return (player_1.options, player_2.options)


    #pick a possible move based on the odd or even player
    def state_transition(self, isOdd):
        """Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]"""
        # shuffle the set of allowed options, pop the first from the resultant list to be used for the move.
        # in this setup, one players plays odd numbers and the other player plays even numbers, 
        # so it is required to switch between the two to determine the next move pick
        
        if(isOdd):
            self.player_1.options = random.sample(self.player_1.options, len(self.player_1.options))
            return self.player_1.options.pop()

        else:
            self.player_2.options = random.sample(self.player_2.options, len(self.player_2.options))
            return self.player_2.options.pop()


    def step(self, isOdd, move):
    #take next step and return reward
        self.board[move-1]= self.state_transition(isOdd)
        reward, done = self.is_winning()
        return reward, done
        # "reward" would be 0 or 1. "done" would be True if game is terminal that is, it ends with win/lose/tie. It would be False otherwise.
    
    #begin training
    def startTraining(self, player_1, player_2, iterations, odd=True, verbose = False):
        self.player_1=player_1
        self.player_2=player_2
        print ("Training Started")
        for i in range(iterations):
            if verbose: print("training ", i)
            self.player_1.game_begin()
            self.player_2.game_begin()
            self.reset()
            done = False

            # Odd player always begins the game for this simulation hence, it is set to true by default
            isOdd = odd
            while not done:
                if isOdd:
                    move = self.player_1.epsilon_greedy(self.board, self.allowed_moves())
                else:
                    move = self.player_2.epsilon_greedy(self.board, self.allowed_moves())
                reward, done = self.step(isOdd, move)
                   
                # Handling the cases where game is terminal or not
                if (reward == 1):  # decisive Win (for 1 agent, the other loses obviously), reward 10 for the winning agent and -10 to losing agent
                    if (isOdd):
                        self.player_1.updateQ(10, self.board, self.allowed_moves())
                        self.player_2.updateQ(-10, self.board, self.allowed_moves())
                    else:
                        self.player_1.updateQ(-10, self.board, self.allowed_moves())
                        self.player_2.updateQ(10, self.board, self.allowed_moves())
                elif (done == False):  # a move was made but game has not ended yet,  reward -1 for the move
                    if (isOdd):
                        self.player_1.updateQ(-1, self.board, self.allowed_moves())
                   

                else: #Tie case
                    self.player_1.updateQ(reward, self.board, self.allowed_moves())
                    self.player_2.updateQ(reward, self.board, self.allowed_moves())


                isOdd = not isOdd  # switching the odd and even agents
        print ("Training Completed")

    #save the Q-tables
    def saveStates(self):
        self.player_1.saveQ("oddPolicy")
        self.player_2.saveQ("evenPolicy")

    #return the Q-tables
    def getQ(self):
        return self.player_1.Q, self.player_2.Q