"""Tabular SARSA and Q-learning update rules."""
import numpy as np
from .environment import N_ACTIONS


def epsilon_greedy(Q, state_id, epsilon, rng=None):
    """Choose a random action with probability epsilon; otherwise greedy.

    Random tie-breaking avoids always preferring the lowest action ID when
    multiple actions have the same maximum Q-value.
    """
    rng = np.random.default_rng() if rng is None else rng

    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))

    values = Q[state_id]
    best_actions = np.flatnonzero(np.isclose(values, np.max(values)))
    return int(rng.choice(best_actions))


def sarsa_update(Q, s, a, reward, next_s, next_a, done, alpha, gamma):
    """On-policy SARSA update."""
    target = reward if done else reward + gamma * Q[next_s, next_a]
    Q[s, a] += alpha * (target - Q[s, a])
    return Q


def q_learning_update(Q, s, a, reward, next_s, done, alpha, gamma):
    """Off-policy Q-learning update."""
    target = reward if done else reward + gamma * np.max(Q[next_s])
    Q[s, a] += alpha * (target - Q[s, a])
    return Q
