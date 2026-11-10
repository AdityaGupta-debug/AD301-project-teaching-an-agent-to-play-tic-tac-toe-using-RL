import unittest
from environment import TicTacToe

class TestTicTacToe(unittest.TestCase):
    def test_win(self):
        game = TicTacToe()
        game.step(0, 1)
        game.step(3, -1)
        game.step(1, 1)
        game.step(4, -1)
        _, done, winner = game.step(2, 1)
        self.assertTrue(done)
        self.assertEqual(winner, 1)

    def test_draw(self):
        game = TicTacToe()
        # Draw sequence: X O X / X O O / O X X
        moves = [(0,1),(1,-1),(2,1),(4,-1),(3,1),(6,-1),(5,1),(8,-1),(7,1)]
        for action, player in moves:
            _, done, winner = game.step(action, player)
        self.assertTrue(done)
        self.assertEqual(winner, 0)

    def test_illegal_move(self):
        game = TicTacToe()
        game.step(0, 1)
        with self.assertRaises(ValueError):
            game.step(0, -1)

if __name__ == "__main__":
    unittest.main()

