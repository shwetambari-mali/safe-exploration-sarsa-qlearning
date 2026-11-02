"""Training loops and training-history collection."""
import numpy as np

from .environment import START, state_to_id, step
from .algorithms import epsilon_greedy, sarsa_update, q_learning_update


def _train(
    algorithm,
    num_episodes=2000,
    alpha=0.1,
    gamma=0.95,
    epsilon=0.2,
    max_steps=100,
    seed=0,
):
    if algorithm not in {"sarsa", "q_learning"}:
        raise ValueError("algorithm must be 'sarsa' or 'q_learning'")
    if not (0 <= epsilon <= 1):
        raise ValueError("epsilon must be between 0 and 1")

    rng = np.random.default_rng(seed)
    Q = np.zeros((36, 4), dtype=float)

    rewards_history = []
    steps_history = []
    hazards_history = []
    success_history = []

    for _ in range(num_episodes):
        state = START
        s = state_to_id(state)
        total_reward = 0.0
        hazard_count = 0
        success = False

        action = epsilon_greedy(Q, s, epsilon, rng)

        for t in range(max_steps):
            next_state, reward, done = step(state, action)
            next_s = state_to_id(next_state)

            if algorithm == "sarsa":
                # SARSA uses the next action actually selected by the policy.
                next_action = epsilon_greedy(Q, next_s, epsilon, rng)
                sarsa_update(
                    Q, s, action, reward, next_s, next_action,
                    done, alpha, gamma
                )
            else:
                # Q-learning uses max_a Q(next_state, a), not next_action.
                q_learning_update(
                    Q, s, action, reward, next_s, done, alpha, gamma
                )
                next_action = None

            total_reward += reward
            if reward < -1:
                hazard_count += 1

            if done:
                success = True
                break

            state = next_state
            s = next_s

            if algorithm == "sarsa":
                action = next_action
            else:
                action = epsilon_greedy(Q, s, epsilon, rng)

        rewards_history.append(total_reward)
        steps_history.append(t + 1)
        hazards_history.append(hazard_count)
        success_history.append(success)

    history = {
        "rewards": np.asarray(rewards_history, dtype=float),
        "steps": np.asarray(steps_history, dtype=int),
        "hazards": np.asarray(hazards_history, dtype=int),
        "successes": np.asarray(success_history, dtype=bool),
    }
    return Q, history


def train_sarsa(num_episodes=2000, alpha=0.1, gamma=0.95,
                epsilon=0.2, max_steps=100, seed=0):
    """Train SARSA and return (Q_table, history)."""
    return _train("sarsa", num_episodes, alpha, gamma, epsilon, max_steps, seed)


def train_q_learning(num_episodes=2000, alpha=0.1, gamma=0.95,
                     epsilon=0.2, max_steps=100, seed=0):
    """Train Q-learning and return (Q_table, history)."""
    return _train("q_learning", num_episodes, alpha, gamma, epsilon, max_steps, seed)
