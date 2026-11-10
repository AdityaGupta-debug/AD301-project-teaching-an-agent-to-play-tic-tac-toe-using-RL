"""Train the Q-learning Tic-Tac-Toe agent."""
import argparse
import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent import QLearningAgent
from environment import TicTacToe

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "q_table.pkl"
OUTPUT_DIR = ROOT / "outputs"

def train(episodes=50000, seed=42, report_every=1000):
    random.seed(seed)
    agent = QLearningAgent(seed=seed)
    env = TicTacToe()
    history = []
    recent = []

    for episode in range(1, episodes + 1):
        env.reset()
        learner_player = 1 if episode % 2 else -1
        current_player = 1
        transitions = []
        done = False
        winner = 0

        while not done:
            state = env.state()
            legal = env.legal_actions()

            if current_player == learner_player:
                action = agent.choose_action(state, legal, player=learner_player, explore=True)
                next_state, done, winner = env.step(action, current_player)
                transitions.append((state, action, next_state, done))
            else:
                action = random.choice(legal)
                _, done, winner = env.step(action, current_player)

            current_player *= -1

        reward = 1.0 if winner == learner_player else (-1.0 if winner == -learner_player else 0.0)


        # Update the learner's decisions backwards. Intermediate rewards are
        # zero; only the final transition receives the terminal outcome.
        for index in range(len(transitions) - 1, -1, -1):
            state, action, next_state, terminal = transitions[index]
            is_last = index == len(transitions) - 1
            step_reward = reward if is_last else 0.0
            # At the next learner decision, the opponent has made one move.
            # Use the next recorded learner state as the bootstrap state.
            if not is_last:
                bootstrap_state = transitions[index + 1][0]
                next_legal = [a for a, v in enumerate(bootstrap_state) if v == 0]
            else:
                bootstrap_state = next_state
                next_legal = []
            agent.update(state, action, step_reward, bootstrap_state,
                         next_legal, player=learner_player, done=is_last)

        agent.decay_epsilon()
        recent.append(1 if winner == learner_player else (0 if winner == 0 else -1))

        if episode % report_every == 0 or episode == episodes:
            window = recent[-report_every:]
            wins = window.count(1) / len(window)
            draws = window.count(0) / len(window)
            losses = window.count(-1) / len(window)
            history.append((episode, wins, draws, losses, agent.epsilon))
            print(f"Episode {episode:>7}/{episodes} | "
                  f"win {wins:.3f} | draw {draws:.3f} | "
                  f"loss {losses:.3f} | epsilon {agent.epsilon:.3f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    agent.save(MODEL_PATH)

    csv_path = OUTPUT_DIR / "training_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "win_rate", "draw_rate", "loss_rate", "epsilon"])
        writer.writerows(history)

    episodes_x = [row[0] for row in history]
    plt.figure(figsize=(10, 5))
    plt.plot(episodes_x, [row[1] for row in history], label="Win rate")
    plt.plot(episodes_x, [row[2] for row in history], label="Draw rate")
    plt.plot(episodes_x, [row[3] for row in history], label="Loss rate")
    plt.xlabel("Training episode")
    plt.ylabel("Rate in reporting window")
    plt.title("Q-learning Tic-Tac-Toe: training progress")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    graph_path = OUTPUT_DIR / "training_progress.png"
    plt.savefig(graph_path, dpi=160)
    plt.close()

    print(f"\nSaved Q-table: {MODEL_PATH}")
    print(f"Saved training graph: {graph_path}")
    print(f"Saved training log: {csv_path}")
    return agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-every", type=int, default=1000)
    args = parser.parse_args()
    train(args.episodes, args.seed, args.report_every)
