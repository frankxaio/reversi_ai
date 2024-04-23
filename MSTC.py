import random

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.simulations = 0
        self.untried_moves = getValidMoves(state, state.current_player)

    def ucb1(self):
        exploration_weight = 1.414
        if self.simulations == 0:
            return float('inf')  # to ensure unexplored nodes are prioritized
        return (self.wins / self.simulations) + exploration_weight * (math.log(self.parent.simulations) / self.simulations)

    def select_child(self):
        return max(self.children, key=lambda node: node.ucb1())

    def expand(self):
        move = self.untried_moves.pop()
        new_board = getBoardCopy(self.state)
        makeMove(new_board, new_board.current_player, move[0], move[1])
        child_node = MCTSNode(new_board, self, move)
        self.children.append(child_node)
        return child_node

    def simulate(self):
        current_simulation_state = getBoardCopy(self.state)
        current_player = current_simulation_state.current_player
        while not isGameOver(current_simulation_state):
            possible_moves = getValidMoves(current_simulation_state, current_player)
            if not possible_moves:
                break
            move = random.choice(possible_moves)
            makeMove(current_simulation_state, current_player, move[0], move[1])
            current_player = 'black' if current_player == 'white' else 'white'
        return current_simulation_state

    def backpropagate(self, simulation_result):
        # In the context of this game, fewer pieces are better.
        current_score = getScoreOfBoard(simulation_result)
        score = current_score[self.parent.state.current_player]
        inverse_score = 64 - score  # Maximizing opponent's score minimizes player's score.
        self.wins += inverse_score
        self.simulations += 1
        if self.parent:
            self.parent.backpropagate(simulation_result)

def monte_carlo_tree_search(root_state, iterations=1000):
    root_node = MCTSNode(root_state)

    for _ in range(iterations):
        node = root_node
        # Selection
        while node.children:
            node = node.select_child()

        # Expansion
        if node.untried_moves:
            node = node.expand()

        # Simulation
        simulation_result = node.simulate()

        # Backpropagation
        node.backpropagate(simulation_result)

    return max(root_node.children, key=lambda n: n.wins / n.simulations).move

# Replace the AI move logic in the game loop with a call to monte_carlo_tree_search
if __name__ == '__main__':
  
