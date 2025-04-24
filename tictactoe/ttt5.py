import numpy as np

class TicTacToeEnv:
    """
    Tic Tac Toe Environment
    - Board is represented as a size x size numpy array (default 5x5)
    - 0 = empty, 1 = X, -1 = O
    - X always goes first
    - Rewards: +1 for win, -1 for loss, 0 for draw or ongoing game
    - Win condition: 5 in a line (horizontally, vertically, or diagonally)
    """
    def __init__(self, state=None, player=1, size=5, win_length=5):
        # Initialize board as empty grid
        self.size = size
        self.win_length = win_length
        
        if state is None: 
            state = np.zeros((size, size), dtype=int)
        self.board = state
        self.current_player = player  # Player 1 (X) starts
        self.done = self.is_done()
        self.action_space = size * size  # Total possible moves
        self.reset()
        
    def is_done(self):
        """Check if the game is finished."""
        if self.check_winner() != 0 or len(self.get_valid_actions()) == 0:
            self.done = True
            return True
        return False
        
    def reset(self):
        """Reset the game to initial state."""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1  # X starts
        self.done = self.is_done()
        return self.get_state(), {}
        
    def get_state(self):
        """Return current board state."""
        return self.board.copy()
        
    def get_valid_actions(self):
        """Return list of valid actions (empty cells)."""
        valid = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == 0:
                    valid.append(i * self.size + j)  # Convert to 1D index
        if valid == []: self.done = True
        return valid
        
    def step(self, action):
        """
        Take action and return new state, reward, done, truncated, info
        action: integer 0-(size*size-1) representing position on the board
        """
        if self.done:
            return self.get_state(), 0, True, False, {"info": "Game already finished"}
            
        # Convert action to 2D position
        row, col = action // self.size, action % self.size
        
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
        # Check for win_length in a row (horizontally, vertically, diagonally)
        
        # Check rows
        for i in range(self.size):
            for j in range(self.size - self.win_length + 1):
                window = self.board[i, j:j+self.win_length]
                if abs(window.sum()) == self.win_length:
                    return self.board[i, j]
                
        # Check columns
        for i in range(self.size - self.win_length + 1):
            for j in range(self.size):
                window = self.board[i:i+self.win_length, j]
                if abs(window.sum()) == self.win_length:
                    return self.board[i, j]
                
        # Check diagonals (top-left to bottom-right)
        for i in range(self.size - self.win_length + 1):
            for j in range(self.size - self.win_length + 1):
                diag = [self.board[i+k, j+k] for k in range(self.win_length)]
                if abs(sum(diag)) == self.win_length:
                    return self.board[i, j]
                
        # Check diagonals (top-right to bottom-left)
        for i in range(self.size - self.win_length + 1):
            for j in range(self.win_length - 1, self.size):
                diag = [self.board[i+k, j-k] for k in range(self.win_length)]
                if abs(sum(diag)) == self.win_length:
                    return self.board[i, j]
                
        return 0
        
    def render(self):
        """Display the board."""
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        horizontal_line = "-" * (self.size * 4 + 1)
        print(horizontal_line)
        for i in range(self.size):
            row = "| "
            for j in range(self.size):
                row += symbols[self.board[i, j]] + " | "
            print(row)
            print(horizontal_line)
        print()

    def human_play(self):
        """Interactive game play for humans using GUI."""
        import tkinter as tk
        from tkinter import messagebox

        self.reset()
        players = {1: "X", -1: "O"}
        
        # Create GUI window
        root = tk.Tk()
        root.title("Tic Tac Toe")
        
        # Configure window
        cell_size = 60
        window_width = cell_size * self.size
        window_height = cell_size * self.size + 40  # Extra space for status
        root.geometry(f"{window_width}x{window_height}")
        
        # Create status label
        status_var = tk.StringVar()
        status_var.set(f"Player {players[self.current_player]}'s turn")
        status = tk.Label(root, textvariable=status_var, height=2)
        status.pack(fill=tk.X)
        
        # Create game board frame
        board_frame = tk.Frame(root)
        board_frame.pack()
        
        # Create buttons for each cell
        buttons = []
        for i in range(self.size):
            row_buttons = []
            for j in range(self.size):
                btn = tk.Button(
                    board_frame, 
                    text=" ", 
                    width=3, 
                    height=1,
                    font=("Arial", 14),
                    command=lambda row=i, col=j: make_move(row, col)
                )
                btn.grid(row=i, column=j)
                row_buttons.append(btn)
            buttons.append(row_buttons)
        
        def make_move(row, col):
            """Handle button click to make a move"""
            if self.done:
                return
                
            action = row * self.size + col
            
            # Check if move is valid
            if action not in self.get_valid_actions():
                return
                
            # Make the move
            _, reward, done, _, info = self.step(action)
            
            # Update the button
            buttons[row][col].config(
                text=players[self.board[row, col]],
                state=tk.DISABLED,
                disabledforeground="black" if self.board[row, col] == 1 else "red"
            )
            
            # Check game state
            if done:
                if "winner" in info and info["winner"] != "Draw":
                    status_var.set(f"Player {info['winner']} wins!")
                    messagebox.showinfo("Game Over", f"Player {info['winner']} wins!")
                else:
                    status_var.set("It's a draw!")
                    messagebox.showinfo("Game Over", "It's a draw!")
                
                # Disable all buttons when game is over
                for r in buttons:
                    for btn in r:
                        btn.config(state=tk.DISABLED)
            else:
                status_var.set(f"Player {players[self.current_player]}'s turn")
        
        def console_play():
            """Switch to console play mode"""
            root.destroy()
            self.reset()
            self._console_play()
        
        def _console_play(self):
            """Original console-based gameplay"""
            self.reset()
            players = {1: "X", -1: "O"}
            
            while not self.done:
                self.render()
                print(f"Player {players[self.current_player]}'s turn")
                
                valid_moves = self.get_valid_actions()
                print(f"Valid moves: {valid_moves}")
                
                try:
                    move = int(input(f"Enter position (0-{self.size*self.size-1}): "))
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
        
        # Add a console mode button (optional)
        console_btn = tk.Button(root, text="Switch to Console Mode", command=console_play)
        console_btn.pack(pady=5)
        
        # Start the game
        root.mainloop()

    def _console_play(self):
        """Original console-based gameplay for compatibility"""
        self.reset()
        players = {1: "X", -1: "O"}
        
        while not self.done:
            self.render()
            print(f"Player {players[self.current_player]}'s turn")
            
            valid_moves = self.get_valid_actions()
            print(f"Valid moves: {valid_moves}")
            
            try:
                move = int(input(f"Enter position (0-{self.size*self.size-1}): "))
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
        clone_env = TicTacToeEnv(size=self.size, win_length=self.win_length)
        clone_env.board = self.board.copy()
        clone_env.current_player = self.current_player
        clone_env.done = self.done
        return clone_env

# Example usage
if __name__ == "__main__":
    # Create a 5x5 board with 5-in-a-line win condition
    env = TicTacToeEnv(size=25, win_length=5)
    env.human_play()