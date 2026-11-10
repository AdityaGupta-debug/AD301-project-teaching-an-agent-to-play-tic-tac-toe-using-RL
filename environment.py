"""Tic-Tac-Toe environment. Board values: 0=empty, 1=X, -1=O."""
import numpy as np

WIN_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros(9, dtype=np.int8)
        self.done = False
        self.winner = 0
        return self.state()

    def state(self):
        return tuple(int(x) for x in self.board)

    def legal_actions(self):
        return [i for i, value in enumerate(self.board) if value == 0]

    def step(self, action, player):
        if self.done:
            raise ValueError("The game is already over. Call reset().")
        if player not in (-1, 1):
            raise ValueError("player must be 1 (X) or -1 (O)")
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action: {action}")
        self.board[action] = player

        for a, b, c in WIN_LINES:
            if self.board[a] == self.board[b] == self.board[c] == player:
                self.done, self.winner = True, player
                return self.state(), True, player

        if not self.legal_actions():
            self.done, self.winner = True, 0
            return self.state(), True, 0

        return self.state(), False, 0

    def render_text(self):
        symbols = {0: ".", 1: "X", -1: "O"}
        rows = []
        for r in range(3):
            rows.append(" | ".join(symbols[int(v)] for v in self.board[r*3:r*3+3]))
        return "\n---------\n".join(rows)
