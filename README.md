# Tic-Tac-Toe RL Agent

A complete beginner-friendly reinforcement-learning project. It trains a tabular Q-learning agent to play Tic-Tac-Toe, saves the learned Q-table, plots training progress, and visualizes the trained agent playing games.

## Features
- Tabular Q-learning with an epsilon-greedy exploration policy.
- Training against a random opponent, with the agent alternating between X and O.
- Q-table persistence (`models/q_table.pkl`).
- Training metrics saved to CSV and plotted to `outputs/training_progress.png`.
- Pygame visualization: watch the trained agent play against another trained agent or play against it yourself.
- A terminal evaluation script to estimate win/draw/loss rates.

## 1. Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 2. Train the agent

From the project root:
```bash
python train.py
```

This trains the agent for 50,000 episodes by default, saves the Q-table in `models/`, writes episode statistics to `outputs/training_log.csv`, and generates a training graph.

Change the number of episodes:
```bash
python train.py --episodes 100000
```

## 3. Evaluate the trained agent

```bash
python evaluate.py --games 5000
```

The evaluation runs the trained agent against a random opponent, alternating the agent's starting mark.

## 4. Watch the trained agent

Watch two copies of the trained agent play:
```bash
python visualize.py --mode self --games 5
```

Play against the trained agent:
```bash
python visualize.py --mode human
```

Controls in human mode:
- Click an empty square to make your move.
- Press `R` to restart.
- Press `ESC` or close the window to quit.

In self-play mode, press `SPACE` to pause/resume, `N` to step to the next game, and `ESC` to quit.

## Project structure

```text
tic_tac_toe_rl/
├── agent.py              # Q-learning agent and policy
├── environment.py        # Tic-Tac-Toe environment and game rules
├── train.py              # Training loop and metrics
├── evaluate.py           # Win/draw/loss evaluation
├── visualize.py          # Pygame gameplay visualization
├── requirements.txt
├── models/               # Saved Q-table (created after training)
├── outputs/              # CSV and training graph (created after training)
└── tests/
    └── test_game.py
```

## How the RL setup works

- **State:** the 9 board cells, represented as a tuple. Each cell is `0` (empty), `1` (X), or `-1` (O).
- **Action:** choose one of the currently empty cell indices (`0` through `8`).
- **Reward:** `+1` for a win, `0` for a draw, and `-1` for a loss.
- **Algorithm:** tabular Q-learning:
  `Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]`
- **Exploration:** epsilon-greedy. Early in training the agent explores more; epsilon gradually decays so it increasingly uses what it has learned.

The Q-table stores values from the perspective of the learning agent. The opponent is a random legal player. Training alternates whether the learner is X or O to reduce first-player bias.

## Notes for your demonstration

1. Run `python train.py` first.
2. Show `outputs/training_progress.png` to explain learning over episodes.
3. Run `python visualize.py --mode self --games 5` to show the trained policy playing.
4. Run `python visualize.py --mode human` for an interactive demonstration.

The training curve is noisy because the opponent is stochastic. A draw against a random opponent is not a failure: Tic-Tac-Toe is a solved game, and the agent is learning a policy rather than memorizing one fixed sequence.
