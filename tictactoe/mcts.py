from ttt import TicTacToeEnv as TTT
import numpy as np
import math
import random
from ttt5 import TicTacToeEnv as TTT5

def calculate_ucb(node_value, node_visits, parent_visits, exploration_weight=1):
    if node_visits == 0:
        return float('inf')  # Always explore unvisited nodes
    
    # Calculate win rate
    exploitation = node_value / node_visits
    
    # UCB exploration term
    exploration = exploration_weight * math.sqrt(math.log(parent_visits) / node_visits)
    
    return exploitation + exploration


class Node:
    def __init__(self, env, parent, action):
        self.env = env.clone()
        self.parent = parent
        self.action = action
        self.reward = 0
        self.visit = 0
        self.children = {}
        self.untried_action = env.get_valid_actions()
        
    def expand_child(self, action):
        new_env = self.env.clone()
        next_state, _, _, _, _ = new_env.step(action)
        # The current_player has changed after taking the action
        next_node = Node(new_env, self, action)
        self.children[action] = next_node
        self.untried_action.remove(action)
        return next_node
        
    def choose_best_child(self, explore_rate=1):
        best_ucb = float('-inf')
        best_child = None
        
        # If no children, return None
        if not self.children:
            return None
            
        for child in self.children.values():
            # UCB formula balances exploration and exploitation
            ucb = calculate_ucb(child.reward, child.visit, self.visit, exploration_weight=explore_rate)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child
            
        return best_child
    
    
    def rollout(self):
        """Simulate a random game from this state to completion."""
        current_env = self.env.clone()
        
        while not current_env.done:
            possible_actions = current_env.get_valid_actions()
            action = random.choice(possible_actions)
            current_env.step(action)
        
        # Return the result from the perspective of the original player
        winner = current_env.check_winner()
        if winner == self.env.current_player:
            return 1  # Win
        elif winner == -self.env.current_player:
            return -1  # Loss
        return 0  # Draw

    def backpropagate(self, result):
        """Backpropagate the result of a simulation up the tree."""
        self.visit += 1
        self.reward += result
        if self.parent:
            self.parent.backpropagate(-result)  # Switch perspective for parent
            
# def mcts(root, num_simulations=1000):
#     # Check for immediate winning moves first
#     for action in root.untried_action:
#         test_env = root.env.clone()
#         next_state, _, done, _, info = test_env.step(action)
#         if done and info.get("winner") == root.env.current_player:
#             return action

#     for i in range(num_simulations):
#         node = root
        
#         # Selection
#         while not node.is_terminate and len(node.untried_action) == 0 and node.children:
#             node = node.choose_best_child(explore_rate=1.4)
        
#         # Expansion
#         if not node.is_terminate and len(node.untried_action) > 0:
#             action = random.choice(node.untried_action)
#             node = node.expand_child(action)
        
#         # Simulation
#         if not node.is_terminate:
#             reward = node.rollout()
#         else:
#             winner = node.env.check_winner()
#             if winner == root.env.current_player:
#                 reward = 1
#             elif winner == -root.env.current_player:
#                 reward = -1
#             else:
#                 reward = 0
        
#         # Backpropagation
#         node.backpropagate(reward)
    
#     # Exploitation only for final selection
#     return max(root.children.values(), key=lambda child: child.visit).action


def mcts(initial_env, num_simulations=1000):
        """Run MCTS from the given state."""
        root = Node(initial_env, parent=None, action=None)
        
        for _ in range(num_simulations):
            node = root
            
            # Selection: choose best child until leaf node found
            while len(node.untried_action)==0 and not node.env.done:
                node = node.choose_best_child(1.4)
            
            # Expansion: if we can expand (i.e., state/node is non-terminal)
            if not node.env.done:
                node = node.expand_child(random.choice(node.untried_action))
            
            # Simulation: rollout from this node
            result = node.rollout()
            
            # Backpropagation
            node.backpropagate(result)
        
        # # Print win rates for each action
        # print("\nAction win rates:")
        # for child in root.children:
        #     win_rate = child.wins / child.visits
        #     print(f"Action {child.action}: Win rate = {win_rate:.4f} ({child.wins}/{child.visits})")
        
        # Return the most visited child
        #return max(root.children, key=lambda child: child.visits).action
        return root.choose_best_child(0).action  # Return the action of the best child node

def mcts_with_human():
    # Ask user which player they want to be
    choice = input("Do you want to be X (goes first) or O? Enter 'X' or 'O': ").upper()
    
    if choice == 'X':
        human_player = 1
        ai_player = -1
        print("You are X (goes first)")
    else:
        human_player = -1
        ai_player = 1
        print("You are O (goes second)")
    
    # Initialize game
    env = TTT5(size=5, player=1)  # Always start with player 1 (X)
    observ, done = env.reset()
    current_player = 1  # X always goes first
    
    # Game loop
    while not done:
        # Display current board
        print("\nCurrent board:")
        env.render()
        
        if current_player == human_player:
            # Human turn
            print("\nYour turn!")
            valid_actions = env.get_valid_actions()
            
            # Print available actions
            print("Available positions (0-8):")
            print([action for action in valid_actions])
            
            # Get human input
            while True:
                try:
                    action = int(input("Enter your move (0-8): "))
                    if action in valid_actions:
                        break
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Please enter a number between 0-8.")
        else:
            # AI turn
            print("\nAI is thinking...")
            action = mcts(env, num_simulations=10000)  # Pass env directly
            print(f"AI chose position {action}")
        
        # Make the move
        observ, reward, done, truncated, info = env.step(action)
        current_player *= -1  # Switch player
        
    # Display final board
    print("\nFinal board:")
    env.render()
    
    # Show result
    winner = info.get("winner")
    if winner == "X" and human_player == 1:
        print("You win!")
    elif winner == "O" and human_player == -1:
        print("You win!")
    elif winner == 'Draw':
        print("It's a draw!")
    else:
        print("AI wins!")



if __name__ == '__main__':
    
    wins = 0
    lost = 0
    draw = 0
    #mcts_with_human()
    
    for episode in range(100):  # Reduced from 1000 for faster testing

        done = False
        truncated = False
        if episode%2 == 0:
            player = 1
        else:
            player = -1
        env = TTT5(size=5, player=player)
        observ, done = env.reset()
        
        while not done and not truncated:
            board = observ
            action = mcts(env, num_simulations=10000)
            if action is None:
                break
            
            observ, reward, done, truncated, info = env.step(action)
            player *= -1
            
        winner = info.get("winner")
        if winner == "X":
            wins += 1
        elif winner == "O":
            lost += 1
        else:  # Draw
            draw += 1
        
        
        print(f"Episode {episode+1}: Wins: {wins}, Losses: {lost}, Draws: {draw}")