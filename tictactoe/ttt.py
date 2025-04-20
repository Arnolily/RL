import numpy as np

class TicTacToeEnv:
    """
    Tic Tac Toe Environment
    - Board is represented as a 3x3 numpy array
    - 0 = empty, 1 = X, -1 = O
    - X always goes first
    - Rewards: +1 for win, -1 for loss, 0 for draw or ongoing game
    """
    def __init__(self, state=None, player = 1):
        # Initialize board as empty 3x3 grid
        if state is None: state = np.zeros((3, 3), dtype=int)
        self.board = state
        self.current_player = player  # Player 1 (X) starts
        self.done = self.is_done()
        self.action_space = 9  # 9 possible moves (positions 0-8)
        self.reset()
        
    def is_done(self):
        """Check if the game is finished."""
        if self.check_winner() != 0 or len(self.get_valid_actions()) == 0:
            self.done = True
            return True
        return False
        
    def reset(self):
        """Reset the game to initial state."""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # X starts
        self.done = self.is_done()
        return self.get_state(), {}
        
    def get_state(self):
        """Return current board state."""
        return self.board.copy()
        
    def get_valid_actions(self):
        """Return list of valid actions (empty cells)."""

        valid = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    valid.append(i * 3 + j)  # Convert to 1D index
        if valid == []: self.done = True
        return valid
        
    def step(self, action):
        """
        Take action and return new state, reward, done, truncated, info
        action: integer 0-8 representing position on the board
        """
        if self.done:
            return self.get_state(), 0, True, False, {"info": "Game already finished"}
            
        # Convert action to 2D position
        row, col = action // 3, action % 3
        
        # Check if valid move
        if self.board[row, col] != 0:
            return self.get_state(), -1, False, False, {"info": "Invalid move"}
            
        # Make move
        self.board[row, col] = self.current_player
        
        # Check for winner
        winner = self.check_winner()
        reward = 0
        info = {"winner": None}
        
        if winner != 0:
            self.done = True
            info["winner"] = "X" if winner == 1 else "O"
            reward = winner * self.current_player  # +1 if win, -1 if loss
        elif len(self.get_valid_actions()) == 0:
            self.done = True
            info["winner"] = "Draw"
            
        # Switch player if game continues
        if not self.done:
            self.current_player *= -1
            
            
        return self.get_state(), reward, self.done, False, info
        
    def check_winner(self):
        """Check if a player has won. Return 1 for X, -1 for O, 0 for no winner."""
        # Check rows
        for i in range(3):
            if abs(self.board[i, :].sum()) == 3:
                return self.board[i, 0]
                
        # Check columns
        for i in range(3):
            if abs(self.board[:, i].sum()) == 3:
                return self.board[0, i]
                
        # Check diagonals
        if abs(np.trace(self.board)) == 3:
            return self.board[0, 0]
            
        if abs(np.trace(np.fliplr(self.board))) == 3:
            return self.board[0, 2]
            
        return 0
        
    def render(self):
        """Display the board."""
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        print("-----------")
        for i in range(3):
            row = "| "
            for j in range(3):
                row += symbols[self.board[i, j]] + " | "
            print(row)
            print("-----------")
        print()
        
    def human_play(self):
        """Interactive game play for humans."""
        self.reset()
        players = {1: "X", -1: "O"}
        
        while not self.done:
            self.render()
            print(f"Player {players[self.current_player]}'s turn")
            
            valid_moves = self.get_valid_actions()
            print(f"Valid moves: {valid_moves}")
            
            try:
                move = int(input("Enter position (0-8): "))
                if move not in valid_moves:
                    print("Invalid move!")
                    continue
            except ValueError:
                print("Please enter a number!")
                continue
                
            _, reward, done, _, info = self.step(move)
            
            if done:
                self.render()
                if "winner" in info and info["winner"] != "Draw":
                    print(f"Player {info['winner']} wins!")
                else:
                    print("It's a draw!")

    def clone(self):
        """Create a deep copy of the current environment state."""
        clone_env = TicTacToeEnv()
        clone_env.board = self.board.copy()
        clone_env.current_player = self.current_player
        clone_env.done = self.done
        return clone_env




# Example usage
if __name__ == "__main__":
    env = TicTacToeEnv()
    env.human_play()