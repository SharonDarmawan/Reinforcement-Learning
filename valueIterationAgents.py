# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            values = self.values.copy()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue
                possibleActions = self.mdp.getPossibleActions(state)
                maxVal = float('-inf')
                for action in possibleActions:
                    maxVal = max(maxVal, self.computeQValueFromValues(state, action))
                values[state] = maxVal
            self.values = values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        transitionList = self.mdp.getTransitionStatesAndProbs(state, action)
        Qval = 0
        
        for nextState, probability in transitionList:
            reward = self.mdp.getReward(state, action, nextState)
            nextVal = self.getValue(nextState)
            Qval += probability * (reward + (self.discount * nextVal))
        return Qval

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        maxVal, maxAction = float('-inf'), None

        if self.mdp.isTerminal(state):
            return None

        for action in self.mdp.getPossibleActions(state):
            Qval = self.computeQValueFromValues(state, action)
            if Qval > maxVal:
                maxVal = Qval
                maxAction = action
        return maxAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        stateList = self.mdp.getStates()
        for i in range(self.iterations):
            # go back to start of stateList once all states have been updated
            stateIdx = i % len(stateList)
            state = stateList[stateIdx]

            if self.mdp.isTerminal(state):
                continue

            possibleActions = self.mdp.getPossibleActions(state)
            maxVal = float('-inf')
            for action in possibleActions:
                maxVal = max(maxVal, self.computeQValueFromValues(state, action))

            self.values[state] = maxVal

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {}
        # Initialize predecessor of a state as a set
        for state in self.mdp.getStates():
            predecessors[state] = set()

        # Compute predecessors of all states
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            for action in self.mdp.getPossibleActions(state):
                nextStateList = [s for s, p in self.mdp.getTransitionStatesAndProbs(state, action) if p != 0]
                for nextState in nextStateList:
                    predecessors[nextState].add(state)

        #Initialize an empty priority queue
        priorityQueue = util.PriorityQueue()

        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            # Absolute value of the difference between the current value of s in self.values
            # and the highest Q-value across all possible actions from s
            possibleActions = self.mdp.getPossibleActions(state)
            maxQVal = max([self.computeQValueFromValues(state, action) for action in possibleActions])
            diff = abs(self.values[state] - maxQVal)
            # Push into priority queue with priority -diff
            priorityQueue.push(state, -diff)

        for i in range(self.iterations):
            # If priority queue is empty, then terminate.
            if priorityQueue.isEmpty():
                break

            # Pop a state s off the priority queue.
            state = priorityQueue.pop()
            # Update the value of s (if it is not a terminal state) in self.values.
            if self.mdp.isTerminal(state):
                continue
            possibleActions = self.mdp.getPossibleActions(state)
            maxVal = float('-inf')
            for action in possibleActions:
                maxVal = max(maxVal, self.computeQValueFromValues(state, action))
            self.values[state] = maxVal

            for predecessor in predecessors[state]:
                # Absolute value of the difference between the current value of p in self.values
                # and the highest Q-value across all possible actions from p
                possibleActions = self.mdp.getPossibleActions(predecessor)
                maxQVal = max([self.computeQValueFromValues(predecessor, action) for action in possibleActions])
                diff = abs(self.values[predecessor] - maxQVal)

                # If diff > theta, push p into the priority queue with priority -diff
                if diff > self.theta:
                    priorityQueue.update(predecessor, -diff)
