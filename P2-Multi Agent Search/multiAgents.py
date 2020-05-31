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
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        min_dist_ghost=1000000
        for ghost in newGhostStates :
            x,y=ghost.getPosition()
            if ghost.scaredTimer ==0 :
                min_dist_ghost=min(min_dist_ghost,manhattanDistance((x,y),newPos))
        min_dist_food=1000000
        for food in newFood.asList():
            min_dist_food=min(min_dist_food,manhattanDistance(food,newPos))


        return -5/(min_dist_ghost+0.1)+5/(min_dist_food +0.1) +min(newScaredTimes) + successorGameState.getScore()

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
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax. 

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState,self.index,self.depth)[1]

    def agents(self,gameState):
        return range(gameState.getNumAgents())
    
    def minimax(self, gameState, agentIndex, depth):
        if depth==0 or gameState.isWin() or gameState.isLose() :
            return (self.evaluationFunction(gameState) , Directions.STOP)
        if agentIndex==0:
            return self.max_value(gameState,agentIndex,depth)
        else:
            return self.min_value(gameState, agentIndex,depth)
              

    def max_value(self, gameState,agentIndex, depth):
        v=-1000000
        best_action=Directions.STOP
        if self.agents(gameState)[-1]==0:
            depth=depth-1
            agent=agentIndex
        else:
            agent=agentIndex+1
        for action in gameState.getLegalActions(agentIndex):
            if self.minimax(gameState.generateSuccessor(agentIndex,action),agent,depth)[0]>v:
                v=self.minimax(gameState.generateSuccessor(agentIndex,action),agent,depth)[0]
                best_action=action
        return (v , best_action)
        
    def min_value(self, gameState, agentIndex ,depth):
        v=1000000
        best_action=Directions.STOP
        if self.agents(gameState)[-1]==agentIndex:
            agent=0
            depth=depth-1
        else:
            agent=agentIndex+1   
        for action in gameState.getLegalActions(agentIndex):
            if  self.minimax(gameState.generateSuccessor(agentIndex,action),agent,depth)[0]<v:
                v= self.minimax(gameState.generateSuccessor(agentIndex,action),agent,depth)[0]
                best_action=action
        return (v , best_action)
   

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def agents(self,gameState):
        return range(gameState.getNumAgents())

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.alphabeta(gameState,self.index,self.depth,-1000000,1000000)[1]
        util.raiseNotDefined()
    
    def alphabeta(self,gameState,agentIndex,depth,alpha,beta):
        if depth==0 or gameState.isLose() or gameState.isWin():
            return (self.evaluationFunction(gameState),Directions.STOP)
        if agentIndex==0:
            return self.alphabeta_max(gameState,agentIndex,depth,alpha,beta)
        else:
            return self.alphabeta_min(gameState,agentIndex,depth,alpha,beta)
    
    def alphabeta_max(self,gameState,agentIndex,depth,alpha,beta):
        if self.agents(gameState)[-1]==0:
            agent=0
            depth=depth-1
        else:
            agent=1
        v=-1000000
        best_action=Directions.STOP
        for action in gameState.getLegalActions(agentIndex):
            x=self.alphabeta(gameState.generateSuccessor(agentIndex,action),agent,depth,alpha,beta)[0]
            if x>v:
                v=x
                best_action=action
                if v> beta:
                    return (v,best_action)
                alpha=max(alpha,v)
        return (v,best_action)

    def alphabeta_min(self, gameState, agentIndex ,depth,alpha,beta):
        v=1000000
        best_action=Directions.STOP
        if agentIndex==self.agents(gameState)[-1]:
            agent=0
            depth=depth-1
        else:
            agent=agentIndex+1   
        for action in gameState.getLegalActions(agentIndex):
            x=self.alphabeta(gameState.generateSuccessor(agentIndex,action),agent,depth,alpha,beta)[0]
            if  x<v:
                v=x
                best_action=action
                if v < alpha:
                    return (v, best_action)
                beta=min(v,beta)
        return (v , best_action)




class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def agents(self,gameState):
        return range(gameState.getNumAgents())

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        return self.expectimax(gameState,self.index,self.depth)[1]
    
    def expectimax(self, gameState, agentIndex, depth):
        if depth==0 or gameState.isWin() or gameState.isLose() :
            return (self.evaluationFunction(gameState) , Directions.STOP)
        if agentIndex==0:
            return self.max_value(gameState,agentIndex,depth)
        else:
            return self.expect_ghosts(gameState, agentIndex,depth)
              

    def max_value(self, gameState,agentIndex, depth):
        v=-1000000
        best_action=Directions.STOP
        if self.agents(gameState)[-1]==0:
            depth=depth-1
            agent=agentIndex
        else:
            agent=agentIndex+1
        for action in gameState.getLegalActions(agentIndex):
            if self.expectimax(gameState.generateSuccessor(agentIndex,action),agent,depth)[0]>v:
                v=self.expectimax(gameState.generateSuccessor(agentIndex,action),agent,depth)[0]
                best_action=action
        return (v , best_action) 

    def expect_ghosts(self, gameState, agentIndex ,depth):
        score=0
        best_action=Directions.STOP
        if self.agents(gameState)[-1]==agentIndex:
            agent=0
            depth=depth-1
        else:
            agent=agentIndex+1   
        for action in gameState.getLegalActions(agentIndex):
            score+= self.expectimax(gameState.generateSuccessor(agentIndex,action),agent,depth)[0]
        v=score/len(gameState.getLegalActions(agentIndex))
        return (v , best_action) 

    

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: We used the same function as for Q1.
    We chose the min distance to a ghost and to a food pellet as main characteristics
    (in Piazza the GSI suggested to do so)
    along with the score and the max scared Time.
    We chose the coefficients by trying with random numbers and 
    tweaking it along the way
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    min_dist_ghost=1000000
    for ghost in newGhostStates :
        (x,y)=ghost.getPosition()
        if ghost.scaredTimer ==0 :
                min_dist_ghost=min(min_dist_ghost,manhattanDistance((x,y),newPos))
    min_dist_food=1000000
    for food in newFood.asList():
        min_dist_food=min(min_dist_food,manhattanDistance(food,newPos))

    return 5/(min_dist_ghost+0.1)+7/(min_dist_food +0.1) +max(newScaredTimes) + currentGameState.getScore()
    

# Abbreviation
better = betterEvaluationFunction
