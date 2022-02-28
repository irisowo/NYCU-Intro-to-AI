# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        return childGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent
    """
    """
    Returns the minimax action from the current gameState using self.depth
    and self.evaluationFunction.

    Here are some method calls that might be useful when implementing minimax.

    gameState.getLegalActions(agentIndex):
    Returns a list of legal actions for an agent
    agentIndex=0 means Pacman, ghosts are >= 1

    gameState.getNextState(agentIndex, action):
    Returns the child game state after an agent takes an action

    gameState.getNumAgents():
    Returns the total number of agents in the game

    gameState.isWin():
    Returns whether or not the game state is a winning state

    gameState.isLose():
    Returns whether or not the game state is a losing state
    """ 
    def getAction(self, gameState):
        # Begin your code (Part 1)
        def minmax(gameState, agentIndex, cur_depth):
            #---------------- terminal state ----------------
            if gameState.isWin() or gameState.isLose() or cur_depth == self.depth :
                return self.evaluationFunction(gameState), None
            #------------------ pre-defined -----------------
            next_agentIndex = (agentIndex + 1) % gameState.getNumAgents()
            next_depth = (cur_depth + 1) if (next_agentIndex == 0) else cur_depth
            #-------------------- Pacman --------------------
            if(agentIndex==0):
                # default value, action
                max_value = float("-inf")
                max_action = None
                # Loop through legal actions to get the maximal value
                # For each action, we can get a next_state and next_state's value
                # If next_state's value is the larger than other past states's value, save it
                for action in gameState.getLegalActions(agentIndex):
                    next_state = gameState.getNextState(agentIndex, action)
                    next_value, _ = minmax(next_state, next_agentIndex, cur_depth) 
                    if next_value > max_value:
                        max_value = next_value
                        max_action = action
                return max_value, max_action
            #-------------------- Ghost --------------------
            else:
                min_value = float("inf")
                min_action = None
                for action in gameState.getLegalActions(agentIndex):
                    next_state = gameState.getNextState(agentIndex, action)
                    next_value, _ = minmax(next_state, next_agentIndex ,next_depth)
                    if next_value < min_value:
                        min_value = next_value
                        min_action = action
                return min_value, min_action
        #---------------------------
        agentIndex, cur_depth = 0, 0
        _, action = minmax(gameState, agentIndex , cur_depth)
        return action
         # End your code (Part 1)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning
    """
    def getAction(self, gameState):
        agentIndex = 0
        cur_depth = 0
        _, action = self.max_value(gameState, agentIndex, cur_depth, float('-inf'),float('inf'))
        return action

    def max_value(self, gameState, agentIndex, cur_depth, alpha, beta):
        # terminal states
        if gameState.isWin() or gameState.isLose() or cur_depth == self.depth :
            return self.evaluationFunction(gameState), None
        # pre-defined next_state's info
        next_agentIndex = (agentIndex + 1) % gameState.getNumAgents()
        next_depth = (cur_depth + 1) if (next_agentIndex == 0) else cur_depth
        # loop through actions to get the max_value
        max_value = float("-inf")
        max_action = None
        for action in gameState.getLegalActions(agentIndex):
            next_state = gameState.getNextState(agentIndex, action) 
            # if next_state is Pacman's turn, call max_value() to get next_state's value
            if(next_agentIndex==0):
                next_value, _ = self.max_value(next_state, next_agentIndex, next_depth, alpha, beta) 
            # if next_state is Ghost's turn, call min_value() to get next_state's value
            else:
                next_value, _ = self.min_value(next_state, next_agentIndex, next_depth, alpha, beta) 
            # update max_value
            if next_value > max_value:
                max_value = next_value
                max_action = action
            # update alpha
            alpha = max(alpha, max_value)
            # Pruning: if max_value > Beta
            if max_value > beta:
                return max_value, action
        return max_value, max_action

    def min_value(self, gameState, agentIndex, cur_depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or cur_depth == self.depth :
            return self.evaluationFunction(gameState), None

        next_agentIndex = (agentIndex + 1) % gameState.getNumAgents()
        next_depth = (cur_depth + 1) if (next_agentIndex == 0) else cur_depth
        # loop through actions to get the min_value
        min_value = float("inf")
        min_action = None
        for action in gameState.getLegalActions(agentIndex):
            next_state = gameState.getNextState(agentIndex, action)
            # if next_state is Pacman's turn, call max_value() to get next_state's value
            if(next_agentIndex==0):
                next_value, _ = self.max_value(next_state, next_agentIndex, next_depth, alpha, beta) 
            # if next_state is Ghost's turn, call min_value() to get next_state's value
            else:
                next_value, _ = self.min_value(next_state, next_agentIndex, next_depth, alpha, beta) 
            # update min_value
            if next_value < min_value:
                min_value = next_value
                min_action = action
            # update beta
            beta = min(beta, min_value)
            # Pruning: if min_value < alpha
            if min_value < alpha:
                return min_value, action
        return min_value, min_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent
    """
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        agentIndex = 0
        cur_depth = 0
        _, action = self.expectimax(gameState, agentIndex, cur_depth)
        return action

    def expectimax(self, game_state, agentIndex, depth):
        #---------------- terminal state ----------------
        if game_state.isWin() or game_state.isLose() or depth == self.depth :
            return self.evaluationFunction(game_state), None
        #------------------ pre-defined ------------------
        next_agentIndex = (agentIndex + 1) % game_state.getNumAgents()
        next_depth = (depth + 1) if (next_agentIndex == 0) else depth
        LegalActions = game_state.getLegalActions(agentIndex)
        #-------------------- Pacman --------------------
        if agentIndex == 0:
            max_value = float("-inf")
            max_action = None

            for action in LegalActions:
                next_state = game_state.getNextState(agentIndex, action)
                next_value, _ = self.expectimax(next_state, next_agentIndex, next_depth)
                if next_value > max_value:
                    max_value = next_value
                    max_action = action
            return max_value, max_action
        #-------------------- Ghost --------------------
        else:
            expected_value = 0
            expected_action = None
            probability = 1.0 / len(LegalActions)

            for action in LegalActions:
                next_state = game_state.getNextState(agentIndex, action)
                next_value, _ = self.expectimax(next_state, next_agentIndex, next_depth)
                expected_value += probability * next_value
            return expected_value, expected_action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme evaluation function
    """
    # 2 kinds of possible factors
    min_food_dist = float('inf')
    game_score = currentGameState.getScore()
    
    # the first factor -- min_food_dist
    pacman_pos = currentGameState.getPacmanPosition()
    for food_position in currentGameState.getFood().asList():
        pac_food_dist = manhattanDistance(pacman_pos, food_position)
        if(pac_food_dist < min_food_dist):
            min_food_dist = pac_food_dist

    # modify weights manually
    min_food_dist_weight = 1
    game_score_weight = 1

    # Final_value = Sum(factor_i * weight_i)
    Final_value = 1.0/(1.0+min_food_dist) * min_food_dist_weight  \
                + game_score * game_score_weight 

    return Final_value

# Abbreviation
better = betterEvaluationFunction
