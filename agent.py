"""Tabular Q-learning agent."""
import pickle
import random
from pathlib import Path

class QLearningAgent:
    def __init__(self, alpha=0.15, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.03, epsilon_decay=0.99992, seed=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q = {}
        self.rng = random.Random(seed)

    @staticmethod
    def _key(state, player):
        # Canonicalize the board to the learner's point of view.
        return tuple(int(v) * int(player) for v in state)

    def q_value(self, state, action, player=1):
        return self.q.get((self._key(state, player), int(action)), 0.0)

    def choose_action(self, state, legal_actions, player=1, explore=False):
        if not legal_actions:
            raise ValueError("No legal actions available")
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(legal_actions)
        values = [self.q_value(state, a, player) for a in legal_actions]
        best = max(values)
        best_actions = [a for a, value in zip(legal_actions, values)
                        if abs(value - best) < 1e-12]
        return self.rng.choice(best_actions)

    def update(self, state, action, reward, next_state, next_legal,
               player=1, done=False):
        key = (self._key(state, player), int(action))
        old = self.q.get(key, 0.0)
        if done or not next_legal:
            target = reward
        else:
            # Q-values are from the learner's perspective. The next state
            # is still evaluated from that same perspective.
            target = reward + self.gamma * max(
                self.q_value(next_state, a, player) for a in next_legal
            )
        self.q[key] = old + self.alpha * (target - old)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({
                "q": self.q,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
            }, f)

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        agent = cls(
            alpha=data.get("alpha", 0.15),
            gamma=data.get("gamma", 0.95),
            epsilon=data.get("epsilon", 0.03),
            epsilon_min=data.get("epsilon_min", 0.03),
            epsilon_decay=data.get("epsilon_decay", 0.99992),
        )
        agent.q = data["q"]
        return agent



