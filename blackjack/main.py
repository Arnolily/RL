import gymnasium as gym
import math
import random
import numpy as np

def calculate_ucb(node_value, node_visits, parent_visits, exploration_weight=1.0):

    if node_visits == 0:
        return float('inf')  # Always explore unvisited nodes
    
    exploitation = node_value / node_visits
    exploration = exploration_weight * math.sqrt(2 * math.log(parent_visits) / node_visits)
    
    return exploitation + exploration

class MCTSNode: 
    # In this node, we need state and parents to be defined. It should contain a set of children nodes and untried actions to determin if all potential children nodes are explored.
    # It should also contain the reward and visits to the node. Besides, it should contain functions to see if the condition is terminate.
    # To expand the decision tree, it should also an expand function to expand child nodes. 
    def __init__(self, state, parent=None, action=None):
        self.state = state  # Blackjack state (player sum, dealer card, usable ace)
        self.parent = parent
        self.action = action  # Action that led to this state
        self.children = {}  # Maps actions to child nodes
        self.visits = 0  # Number of times this node was visited
        self.value = 0  # Total reward accumulated through this node
        self.untried_actions = [0, 1]  # Stick (0) or hit (1)
        
        self.is_terminated = (self.state[0]>21) or self.action == 0
    
    def is_fully_expanded(self):
        return len(self.untried_actions)==0
        
    def choose_best_child(self, exploration_weight=1):
        best_ucb=-100
        best_child = None
        for child in self.children.values():
            ucb = calculate_ucb(child.value, child.visits, self.visits, exploration_weight=exploration_weight)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child
                
        return best_child
    
    def simulate_step(self, state, action):
        # Simulate one step in the blackjack game without using gym
        player_sum, dealer_showing, usable_ace = state
        
        if action == 1:  # Hit
            # Draw a card (simulating from infinite deck)
            card = min(10, np.random.randint(1, 14))  # 1-13 where 11,12,13=10
            
            if card == 1:  # Ace
                if player_sum + 11 <= 21:
                    player_sum += 11
                    usable_ace = 1
                else:
                    player_sum += 1
            else:
                player_sum += card
            
            # Check if we bust and have a usable ace
            if player_sum > 21 and usable_ace:
                player_sum -= 10  # Convert ace from 11 to 1
                usable_ace = 0
        
        return (player_sum, dealer_showing, usable_ace)
    
    
    
    def expand(self, action): 
        # This function is to expand child nodes, and it would only need actions so we know which new state we will go to
        next_state = self.simulate_step(self.state, action)
        self.untried_actions.remove(action)
        
        next_node = MCTSNode(next_state, parent=self, action=action)
        
        self.children[action] = next_node

        return next_node
        
        
    def get_reward(self):
        """Calculate reward at terminal states."""
        player_sum, dealer_showing, usable_ace = self.state
        
        # If player busted
        if player_sum > 21:
            return -1
            
        # If player stuck, simulate dealer's turn
        if self.action == 0:
            # Initialize dealer's hand
            dealer_sum = dealer_showing
            dealer_usable_ace = 1 if dealer_showing == 1 else 0
            
            # Dealer hits until 17 or higher
            while dealer_sum < 17:
                card = min(10, np.random.randint(1, 14))
                
                if card == 1:  # Ace
                    if dealer_sum + 11 <= 21:
                        dealer_sum += 11
                        dealer_usable_ace = 1
                    else:
                        dealer_sum += 1
                else:
                    dealer_sum += card
                    
                # Check if dealer busts with usable ace
                if dealer_sum > 21 and dealer_usable_ace:
                    dealer_sum -= 10
                    dealer_usable_ace = 0
                    
            # Compare hands
            if dealer_sum > 21:  # Dealer busts
                return 1
            if dealer_sum > player_sum:
                return -1
            elif dealer_sum < player_sum:
                return 1
            else:
                return 0  # Push (tie)
                
        return 0  # Default for non-terminal states

    # def rollout(self):
    #     node = self
    #     action = None
    #     while True:
    #         if node.state[0]>21 or action == 0:
                
    #             reward = self.get_reward()
    #             return reward

    #         action = random.choice([0,1])
    #         node.state = self.simulate_step(node.state, action)
            
    def rollout(self):
        current_state = self.state
        action = None
        
        while True:
            if current_state[0] > 21 or action == 0:
                original_state = self.state
                self.state = current_state
                reward = self.get_reward()
                self.state = original_state
                return reward

            action = random.choice([0, 1])
            current_state = self.simulate_step(current_state, action)          

    def backpropagate(self, reward):
        """Update node statistics from leaf to root."""
        node = self
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent
            
            
def mcts_search(root, num_simulations=1000):
    """Main MCTS algorithm."""
    for _ in range(num_simulations):
        # Selection
        node = root
        
        while not node.is_terminated and node.is_fully_expanded():
            node = node.choose_best_child()
            
        # Expansion
        if not node.is_terminated and not node.is_fully_expanded():
            action = random.choice(node.untried_actions)
            node = node.expand(action)
            
        # Simulation
        reward = node.rollout()
            
        # Backpropagation
        node.backpropagate(reward)
        
    # Return best action
    return root.choose_best_child(exploration_weight=0).action  # Pure exploitation



if __name__ == '__main__':
    env = gym.make('Blackjack-v1', natural=False, sab=False)
    wins = 0
    lost = 0
    draw = 0
    for episode in range(1000):
        observ, info = env.reset()
        done = False
        truncated = False
        
        while not done and not truncated:
            state = observ
            action = mcts_search(root=MCTSNode(state), num_simulations=1000)
            
            if action == None:
                break
            
            observ, reward, done, truncated, info = env.step(action)
            
        if reward > 0: wins+=1
        if reward < 0: lost+=1
        if reward == 0: draw +=1
        
        env.close()
        
        print(wins, lost, draw)