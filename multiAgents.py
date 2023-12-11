from operator import itemgetter
import time
import traceback
from Agents import Agent
import util
import random

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    def __init__(self, *args, **kwargs) -> None:
        self.index = 0 # your agent always has index 0

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        It takes a GameState and returns a tuple representing a position on the game board.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(self.index)

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (Game.py) and returns a number, where higher numbers are better.
        You can try and change this evaluation function if you want but it is not necessary.
        """
        nextGameState = currentGameState.generateSuccessor(self.index, action)
        return nextGameState.getScore(self.index) - currentGameState.getScore(self.index)


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    Every player's score is the number of pieces they have placed on the board.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore(0)


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (Agents.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2', **kwargs):
        self.index = 0 # your agent always has index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent which extends MultiAgentSearchAgent and is supposed to be implementing a minimax tree with a certain depth.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

        

    def getAction(self, state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        But before getting your hands dirty, look at these functions:

        gameState.isGameFinished() -> bool
        gameState.getNumAgents() -> int
        gameState.generateSuccessor(agentIndex, action) -> GameState
        gameState.getLegalActions(agentIndex) -> list
        self.evaluationFunction(gameState) -> float
        
        """
        def minMaxValue(currentState,index):
            score = None
            action = None

            def getAgentIndex():
                return index%currentState.getNumAgents()

            def getDepth():
                return int(index/currentState.getNumAgents())

            def cutoffTest():
                return currentState.isGameFinished() or getDepth() >= self.depth
                
            def isMaxAgent(agentIndex):
                return agentIndex == 0
            
            if(cutoffTest()):
                score = self.evaluationFunction(currentState)
                action = None
            else:
                legalActions = currentState.getLegalActions(getAgentIndex())
                successors = [(currentState.generateSuccessor(getAgentIndex(),action),action) for action in legalActions]
                minMaxValues = [(minMaxValue(nextState,index+1),action) for (nextState,action) in successors]
                if(isMaxAgent(getAgentIndex())):
                    (score,action) = max(minMaxValues,key=itemgetter(0))
                else:
                    (score,action) = min(minMaxValues,key=itemgetter(0))
            return (score,action)



        "*** YOUR CODE HERE ***"
        (score,action) = minMaxValue(state,0)
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning. It is very similar to the MinimaxAgent but you need to implement the alpha-beta pruning algorithm too.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        You should keep track of alpha and beta in each node to be able to implement alpha-beta pruning.
        """
        def minMaxValueWithAlphaBetaPruning(currentState,index,alpha,beta):
            score = None
            action = None

            def getAgentIndex():
                return index%currentState.getNumAgents()

            def getDepth():
                return int(index/currentState.getNumAgents())

            def cutoffTest():
                return currentState.isGameFinished() or getDepth() >= self.depth
                
            def isMaxAgent(agentIndex):
                return agentIndex == 0
            
            def maxValue(currentState, alpha,beta):
                value =  -1111111111
                action = None
                legalActions = currentState.getLegalActions(getAgentIndex())
                for _action in legalActions:
                    successor = currentState.generateSuccessor(getAgentIndex(),_action)
                    _value = max(value, minMaxValueWithAlphaBetaPruning(successor,index+1,alpha,beta)[0])
                    if(value != _value):
                        value = _value
                        action = _action 
                    if(value > beta ):
                        break
                    alpha = max(value, alpha)
                return (value,action)


            def minValue(currentState, alpha,beta):
                value =  +1111111111
                action = None
                legalActions = currentState.getLegalActions(getAgentIndex())                
                for _action in legalActions:
                    successor = currentState.generateSuccessor(getAgentIndex(),_action)
                    _value = min(value, minMaxValueWithAlphaBetaPruning(successor,index+1,alpha,beta)[0])
                    if(value != _value):
                        value = _value
                        action = _action 
                    beta = min(value, beta)
                return (value,action)

            if(cutoffTest()):
                score = self.evaluationFunction(currentState)
                action = None
            else:
                if(isMaxAgent(getAgentIndex())):
                    (score,action) = maxValue(currentState,alpha,beta)
                else:
                    (score,action) = minValue(currentState,alpha,beta)

            return (score,action)


        "*** YOUR CODE HERE ***"
        (score,action) = minMaxValueWithAlphaBetaPruning(gameState,0,-1111111111,+1111111111)
        return action
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent which has a max node for your agent but every other node is a chance node.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        You should keep track of alpha and beta in each node to be able to implement alpha-beta pruning.
        """
        def exeptiMiniMax(currentState,index):
            score = None
            action = None

            def getAgentIndex():
                return index%currentState.getNumAgents()

            def getDepth():
                return int(index/currentState.getNumAgents())

            def cutoffTest():
                return currentState.isGameFinished() or getDepth() >= self.depth
                
            def isMaxAgent(agentIndex):
                return agentIndex == 0
            
            def maxValue(currentState):
                value =  -1111111111
                action = None
                legalActions = currentState.getLegalActions(getAgentIndex())
                for _action in legalActions:
                    successor = currentState.generateSuccessor(getAgentIndex(),_action)
                    _value = max(value, exeptiMiniMax(successor,index+1)[0])
                    if(value != _value):
                        value = _value
                        action = _action 
                return (value,action)


            def minValue(currentState):
                legalActions = currentState.getLegalActions(getAgentIndex())        
                successors = [(currentState.generateSuccessor(getAgentIndex(),_action),_action) for _action in legalActions]
                scores = [(exeptiMiniMax(nextState,index+1)[0],_action) for (nextState,_action) in successors]
                (value, action) = random.choice(scores)
                return (value,action)

            if(cutoffTest()):
                score = self.evaluationFunction(currentState)
                action = None
            else:
                if(isMaxAgent(getAgentIndex())):
                    (score,action) = maxValue(currentState)
                else:
                    (score,action) = minValue(currentState)

            return (score,action)


        "*** YOUR CODE HERE ***"
        (score,action) = exeptiMiniMax(gameState,0)
        return action


def betterEvaluationFunction(currentGameState):
    """
    Your extreme evaluation function.

    You are asked to read the following paper on othello heuristics and extend it for two to four player rollit game.
    Implementing a good stability heuristic has extra points.
    Any other brilliant ideas are also accepted. Just try and be original.

    The paper: Sannidhanam, Vaishnavi, and Muthukaruppan Annamalai. "An analysis of heuristics in othello." (2015).

    Here are also some functions you will need to use:
    
    gameState.getPieces(index) -> list
    gameState.getCorners() -> 4-tuple
    gameState.getScore() -> list
    gameState.getScore(index) -> int

    """
    
    "*** YOUR CODE HERE ***"

    # parity
    def parityHeuristicFunction():
        maxScore = currentGameState.getScore(0)
        minScores = sum(currentGameState.getScore()[1:])
        _sum = maxScore + minScores
        _minus = maxScore - minScores
        return ((_minus/_sum))*100

    # corners
    def cornersHeuristicFunction():
        corners = currentGameState.getCorners()
        maxCorners = 0
        minCorners = 0
        result = 0
        for cornerValue in corners:
            if(cornerValue == 0):
                maxCorners =+1
            elif(cornerValue>0):
                minCorners =+1
        _sum = maxCorners + minCorners
        _minus = maxCorners - minCorners
        if(_sum != 0):
            result = ((_minus/_sum))*100
        return result


    # mobility
    def mobilityHeuristicFunction():
        agentsMobilities = [len(currentGameState.getLegalActions(i)) for i in range(0,currentGameState.getNumAgents())]
        maxMobility = agentsMobilities[0]
        result = 0
        minMobilities = sum(agentsMobilities[1:])
        _sum = maxMobility + minMobilities
        _minus = maxMobility - minMobilities
        if(_sum != 0):
            result = ((_minus/_sum))*100
        return result
    # stability
    def stabilityHeuristicFunction():
        def getAgentStablityValues():
            def getAgentStablity(agentIndex,successors):
                agentPieces = currentGameState.getPieces(agentIndex)
                semiStables = {}
                stables = set(agentPieces.copy())
                nonStables = set({})
                for nextState in successors:
                    newPieces = set(nextState.getPieces(agentIndex))
                    for (agentPiece) in agentPieces:
                        if not (agentPiece) in newPieces:
                            stables = stables - {agentPiece}
                            if((agentPiece) in semiStables):
                                semiStables[(agentPiece)] =  semiStables[(agentPiece)] + 1
                            else:
                                semiStables[(agentPiece)] = 1
                for key in semiStables.keys():
                    if(semiStables[key] >= int(len(agentPieces)/4))*3:
                        nonStables.add(key)
                return len(stables) - len(nonStables)
            nextPlayerIndex = 1
            result = []
            legalActions = currentGameState.getLegalActions(nextPlayerIndex)
            successors = [currentGameState.generateSuccessor(nextPlayerIndex,action) for action in legalActions]
            for index in range(0,currentGameState.getNumAgents()):
                result.append(getAgentStablity(index,successors))
            return result



        stablities = getAgentStablityValues()
        maxStablity = stablities[0]
        minStablities = sum(stablities[1:])

        result = 0
        _sum = (maxStablity) + (minStablities)
        _minus = maxStablity - minStablities
        if(_sum != 0):
            result = ((_minus/_sum))*100
        return result

    cornersHeuristic = 0
    parityHeuristic = 0
    stabilityHeuristic = 0
    mobilityHeuristic = 0

    numberOfapturedSquares = sum(([len(currentGameState.getPieces(index)) for index in range(0,currentGameState.getNumAgents())]))

    if (numberOfapturedSquares > 16):
        parityHeuristic = 16 * parityHeuristicFunction()
        stabilityHeuristic = 0
    else:
        parityHeuristicFunction = 4 * parityHeuristicFunction()
        stabilityHeuristic = 6 * stabilityHeuristicFunction()

    mobilityHeuristic = 8 * mobilityHeuristicFunction()
    cornersHeuristic = 64 * cornersHeuristicFunction()
    result = cornersHeuristic + parityHeuristic + stabilityHeuristic + mobilityHeuristic
    return  result

# Abbreviation
better = betterEvaluationFunction