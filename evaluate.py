"""Evaluate a saved agent against a random legal opponent."""
import argparse
import random
from pathlib import Path

from agent import QLearningAgent
from environment import TicTacToe

def evaluate(games=5000, seed=123):
    random.seed(seed)
    model = Path(__file__).resolve().parent / "models" / "q_table.pkl"
    if not model.exists():
        raise SystemExit("No trained model found. Run: python train.py")
    agent = QLearningAgent.load(model)
    results = {"win": 0, "draw": 0, "loss": 0}
    env = TicTacToe()

    for game in range(games):
        env.reset()
        learner = 1 if game % 2 == 0 else -1
        player, done, winner = 1, False, 0
        while not done:
            legal = env.legal_actions()
            if player == learner:
                action = agent.choose_action(env.state(), legal, player=learner)
            else:
                action = random.choice(legal)
            _, done, winner = env.step(action, player)
            player *= -1

        if winner == learner:
            results["win"] += 1
        elif winner == 0:
            results["draw"] += 1
        else:
            results["loss"] += 1

    print(f"Games: {games}")
    for key in ("win", "draw", "loss"):
        count = results[key]
        print(f"{key.capitalize():<5}: {count:>5} ({count / games:.1%})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    evaluate(args.games, args.seed)


