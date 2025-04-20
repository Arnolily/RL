from ttt import TicTacToeEnv as TTT
import numpy as np
import math
import random

def calculate_ucb(node_value, node_visits, parent_visits, exploration_weight=1):
    if node_visits == 0:
        return float('inf')  # Always explore unvisited nodes
    
    exploitation = node_value / node_visits
    exploration = exploration_weight * math.sqrt(2 * math.log(parent_visits) / node_visits)
    
    return exploitation + exploration


class Node:
    def __init__(self, state, parent, action, current_player):
        self.state = state
        self.parent = parent
        self.action = action
        self.reward = 0
        self.visit = 0
        self.children = {}
        self.env = TTT(state=state, player=current_player)
        self.env.board = self.state
        self.env.current_player = current_player
        self.is_terminate = self.env.done
        self.untried_action = self.env.get_valid_actions()
        
    def expand_child(self, action):
        new_env = self.env.clone()
        next_state = new_env.step(action)[0]
        next_node = Node(next_state, self, action, new_env.current_player)
        self.children[action] = next_node
        self.untried_action.remove(action)
        return next_node
        
    def choose_best_child(self, explore_rate=1):
        best_ucb = -1e1000
        best_child = None
        for child in self.children.values():
            ucb = calculate_ucb(child.reward, child.visit, self.visit, exploration_weight=explore_rate)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child
            
        return best_child
    
    def rollout(self):
        new_env = self.env.clone()
        
        # If game is already over, return current result
        if new_env.done:
            winner = new_env.check_winner()
            if winner == 0:
                return 0  # Draw
            return winner * new_env.current_player  # Win/loss from current player perspective
        
        # Play random moves until game ends
        while not new_env.done:
            valid_actions = new_env.get_valid_actions()
            
            # If no valid actions, it's a draw
            if not valid_actions:
                return 0
                
            # Choose random action
            action = random.choice(valid_actions)
            
            # Take action
            _, reward, done, _, _ = new_env.step(action)
            
            # If game ended, return reward
            if done:
                return reward
            
        # Safety return (should not reach here)
        return 0
            
    def backpropagate(self, reward):
        node = self
        orig_player = self.env.current_player
        
        while node:
            node.visit += 1
            
            # Adjust reward based on player perspective
            if node.env.current_player != orig_player:
                node.reward += -reward
            else:
                node.reward += reward
                
            node = node.parent
            
def mcts(root, num_simulations=1000):
    for i in range(num_simulations):
        # Selection
        node = root
        while len(node.untried_action) == 0 and not node.is_terminate:
            child = node.choose_best_child(explore_rate=1.4)
            if child is None:  # No children available
                break
            node = child
        
        # Expansion
        if not node.is_terminate and len(node.untried_action) > 0:
            action = random.choice(node.untried_action)
            node = node.expand_child(action)
            
        # Simulation
        reward = node.rollout()
        
        # Backpropagation
        node.backpropagate(reward)
    
    # Choose best action
    best_child = root.choose_best_child(explore_rate=0)
    return best_child.action if best_child else None


if __name__ == '__main__':
    
    wins = 0
    lost = 0
    draw = 0
    
    for episode in range(100):  # Reduced from 1000 for faster testing

        done = False
        truncated = False
        if episode%2 == 0:
            player = 1
        else:
            player = -1
        env = TTT(player=player)
        observ, done = env.reset()
        
        while not done and not truncated:
            board = observ
            action = mcts(root=Node(observ, parent=None, action=None, current_player=player), num_simulations=1000)
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