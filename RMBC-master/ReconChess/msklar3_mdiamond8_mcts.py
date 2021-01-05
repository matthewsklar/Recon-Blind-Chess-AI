import msklar3_mdiamond8_config as config
import msklar3_mdiamond8_chess_helper as helper

class Node():
    def __init__(self, state, sample, color):
        self.state = state
        self.sample = sample
        self.color = sample.turn

        self.edges = []

    def is_leaf(self):
        return len(self.edges) == 0

    def to_string(self):
        # print("node has state", self.state)
        
        for edge in self.edges:
            print("    edge has values: N = {}, W = {}, Q = {}, P = {}, action={}".format(
                edge.data['N'],
                edge.data['W'],
                edge.data['Q'],
                edge.data['P'],
                helper.action_map(edge.action)
            ))

        
        # for edge in self.edges:
        #     edge.out_node.to_string()

class Edge():
    def __init__(self, in_node, out_node, action, prior):
        self.in_node = in_node
        self.out_node = out_node
        self.action = action

        self.data = {
            'N' : 0,        # number of times action a has been taken from state s
            'W' : 0,        # total value of next state
            'Q' : 0,        # mean value of the next state
            'P' : prior,    # prior probability of selecting action a
        }

    def to_string(self):
        print('action:', self.action)
        print('data:', self.data)

class MCTS():
    def __init__(self, root, agent_color):
        self.root = root
        self.tree = {}
        self.agent_color = agent_color
        self.leaf = None

    # Navigate to and select a leaf for expansion
    def select(self):
        node = self.root
        path = []

        while not node.is_leaf():
            action = self.max_action(node.edges)
    
            path.append(action)
            node = action.out_node

        self.leaf = node

        return node, path

    # Update the Monte-Carle Tree based on simulation results
    def backfill(self, v, path):
        for action in path:
            action.data['N'] = action.data['N'] + 1
            action.data['W'] = action.data['W'] + v * self.agent_turn(action.in_node)
            action.data['Q'] = action.data['W'] / action.data['N']
            

    # Select action that maximizes Q + U. Return the edge representing the best action
    def max_action(self, actions):
        max_a = None
        max_QU = None

        for a in actions:
            Q = a.data['Q']
            U = self.calculate_U(a.data['P'], a.data['N'])

            if max_QU == None or Q + U > max_QU:
                max_a = a
                max_QU = Q + U

        return max_a

    # Calculate U value
    # TODO: Might want to update this to include epsilon but must be proportional to this
    def calculate_U(self, P, N):
        return P / (1 + N)

    def addNode(self, node):
        self.tree[node.id] = node

    def agent_turn(self, node):
        return 1 if node.color == self.agent_color else -1

    def to_string(self):
        q = [self.root]

        for x in q:
            x.to_string()
