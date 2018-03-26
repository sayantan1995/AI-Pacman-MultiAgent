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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

        score = successorGameState.getScore()
        foodWeight = 10
        ghostWeight = 10
        foodDistance = [manhattanDistance(newPos,food) for food in newFood.asList()]
        ghostDistance = manhattanDistance(newPos,newGhostStates[0].getPosition())

        if foodDistance:
          update = foodWeight/float(min(foodDistance))
          score = score + update

        if ghostDistance:
          update = ghostWeight/float(ghostDistance)
          score = score - update

        return score


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
        """
        "*** YOUR CODE HERE ***"

        def minimax(state,depth,agent):

          if agent == state.getNumAgents():

            if depth == self.depth:
              return self.evaluationFunction(state)

            else:
              return minimax(state,depth+1,0)

          else:

            if state.getLegalActions(agent):
              successor = (minimax(state.generateSuccessor(agent,action),depth,agent+1) for action in state.getLegalActions(agent))

            else:
              return self.evaluationFunction(state)

          if agent:
            return min(successor) 

          else:
            return max(successor)

        action = max(gameState.getLegalActions(0),key=lambda X: minimax(gameState.generateSuccessor(0,X),1,1))
        return action 



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        def maxAction(state,depth,agent,alpha,beta):

          if depth > self.depth:
            return self.evaluationFunction(state)

          score = 0

          for action in state.getLegalActions(agent):
            successor = minAction(state.generateSuccessor(agent,action),depth,agent+1,alpha,beta)
            score = max(score,successor)

            if beta and score > beta:
              return score

            else:
              alpha = max(score,alpha)

          if not score:
            return scoreEvaluationFunction(state)

          else:
            return score
        

        def minAction(state,depth,agent,alpha,beta):

          if agent == state.getNumAgents():
            return maxAction(state,depth+1,0,alpha,beta)

          score = 0

          for action in state.getLegalActions(agent):
            successor = minAction(state.generateSuccessor(agent,action),depth,agent+1,alpha,beta)

            if score:
              score = min(score,successor)

            else:
              score = successor

            if alpha and score < alpha:
              return score

            else:

              if beta:
                beta = min(score,beta)

              else:
                beta = score

          if not score:
            return self.evaluationFunction(state)

          return score

        score = None
        alpha = None
        beta = None
        best = None

        for action in gameState.getLegalActions(0):
          score = max (score,minAction(gameState.generateSuccessor(0,action),1,1,alpha,beta))

          if not alpha:
            alpha = score
            best = action

          else:
            alpha = max(score,alpha)

            if score > alpha:
              best = action

        return best
    


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax(state,depth,agent):

          if agent == state.getNumAgents():

            if depth == self.depth:
              return self.evaluationFunction(state)

            else:
              return expectimax(state,depth+1,0)

          else:

            if state.getLegalActions(agent):
              successor = (expectimax(state.generateSuccessor(agent,action),depth,agent+1) for action in state.getLegalActions(agent))

            else:
              return self.evaluationFunction(state)

          if agent:
            successorList = list(successor)
            action = (float(sum(successorList))/len(successorList))
            return action

          else:
            return max(successor)


        action = max(gameState.getLegalActions(0),key=lambda X: expectimax(gameState.generateSuccessor(0,X),1,1))
        return action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    score = currentGameState.getScore()
    eatenGhost = 200
    ghostScore = 0
    ghostWeight = 10
    foodWeight = 10


    for ghost in newGhostStates:
      ghostDistance = manhattanDistance(newPos,newGhostStates[0].getPosition())

      if ghostDistance:

        if ghost.scaredTimer:
          update = eatenGhost/float(ghostDistance)
          ghostScore = ghostScore + update

        else:
          update = ghostWeight/ghostDistance
          ghostScore = ghostScore - update

    score = score + ghostScore
    foodDistance = [manhattanDistance(newPos,food) for food in newFood.asList()]

    if foodDistance:
      update = foodWeight/float(min(foodDistance))
      score = score + update

    return score

# Abbreviation
better = betterEvaluationFunction

