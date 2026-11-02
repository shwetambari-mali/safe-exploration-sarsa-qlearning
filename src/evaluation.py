"""Greedy-policy evaluation and plotting helpers."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

from .environment import (
    ROWS, COLS, START, GOAL, HAZARDS, N_STATES,
    state_to_id, step,
)


def greedy_action(Q, state_id, rng=None):
    """Greedy action with random tie-breaking."""
    rng = np.random.default_rng() if rng is None else rng
    values = Q[state_id]
    best_actions = np.flatnonzero(np.isclose(values, np.max(values)))
    return int(rng.choice(best_actions))


def evaluate_agent(Q, num_tests=100, max_steps=100, seed=123):
    """Evaluate without exploratory actions (epsilon = 0)."""
    rng = np.random.default_rng(seed)
    rewards, hazards, lengths, successes = [], [], [], []

    for _ in range(num_tests):
        state = START
        total_reward = 0.0
        hazard_count = 0
        success = False

        for t in range(max_steps):
            s = state_to_id(state)
            action = greedy_action(Q, s, rng)
            next_state, reward, done = step(state, action)

            total_reward += reward
            if reward < -1:
                hazard_count += 1

            state = next_state
            if done:
                success = True
                break

        rewards.append(total_reward)
        hazards.append(hazard_count)
        lengths.append(t + 1)
        successes.append(success)

    return {
        "average_reward": float(np.mean(rewards)),
        "average_hazards": float(np.mean(hazards)),
        "success_rate": float(np.mean(successes) * 100),
        "average_steps": float(np.mean(lengths)),
        "all_rewards": np.asarray(rewards),
        "all_hazards": np.asarray(hazards),
    }


def moving_average(values, window=50):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        raise ValueError("window cannot exceed the number of data points")
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_learning_curves(sarsa_history, q_history, window=50):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(moving_average(sarsa_history["rewards"], window), label="SARSA")
    axes[0].plot(moving_average(q_history["rewards"], window), label="Q-learning")
    axes[0].set_title(f"Episode return ({window}-episode moving average)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(moving_average(sarsa_history["hazards"], window), label="SARSA")
    axes[1].plot(moving_average(q_history["hazards"], window), label="Q-learning")
    axes[1].set_title(f"Hazard encounters ({window}-episode moving average)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Hazards per episode")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_policy(Q, title="Greedy policy"):
    arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    fig, ax = plt.subplots(figsize=(6, 6))

    for r in range(ROWS):
        for c in range(COLS):
            state = (r, c)
            if state == START:
                label = "S"
            elif state == GOAL:
                label = "G"
            elif state in HAZARDS:
                label = "H"
            else:
                label = arrows[int(np.argmax(Q[state_to_id(state)]))]
            ax.text(c, r, label, ha="center", va="center", fontsize=16)

    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.set_xticks(np.arange(-0.5, COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, ROWS, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def print_comparison(sarsa_results, q_results):
    metrics = [
        ("Average evaluation return", "average_reward"),
        ("Average hazards per episode", "average_hazards"),
        ("Success rate (%)", "success_rate"),
        ("Average steps", "average_steps"),
    ]
    print(f"{'Metric':<32} {'SARSA':>12} {'Q-learning':>12}")
    print("-" * 58)
    for label, key in metrics:
        print(f"{label:<32} {sarsa_results[key]:>12.2f} {q_results[key]:>12.2f}")



def get_agent_trajectory(Q, seed=42, max_steps=100):
    """
    Run a trained agent using its greedy policy and record its path.

    Hazard behavior:
    - Agent enters a hazard.
    - Hazard cell is recorded.
    - Agent is reset to START.
    """

    rng = np.random.default_rng(seed)
    state = START

    # Each segment is a continuous part of the trajectory.
    # A new segment begins after the agent is reset by a hazard.
    segments = [[START]]

    hazard_hits = 0
    success = False
    steps = 0

    # Action mapping: up, down, left, right
    action_deltas = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1)
    }

    for _ in range(max_steps):

        state_id = state_to_id(state)

        # Choose a greedy action.
        # Random tie-breaking avoids always choosing action 0
        # when multiple actions have the same Q-value.
        best_value = np.max(Q[state_id])
        best_actions = np.flatnonzero(
            np.isclose(Q[state_id], best_value)
        )
        action = int(rng.choice(best_actions))

        # Calculate the cell the agent attempts to enter.
        dr, dc = action_deltas[action]
        attempted_state = (state[0] + dr, state[1] + dc)

        next_state, reward, done = step(state, action)
        steps += 1

        if reward < -1:
            # Record the hazard cell before the reset.
            segments[-1].append(attempted_state)
            hazard_hits += 1

            # Start a new trajectory segment after reset.
            if not done:
                segments.append([START])

        else:
            segments[-1].append(next_state)

        state = next_state

        if done:
            success = (state == GOAL)
            break

    return {
        "segments": segments,
        "hazard_hits": hazard_hits,
        "success": success,
        "steps": steps
    }


def plot_agent_trajectory(Q, title="Agent Trajectory", seed=42):
    """
    Visualize the actual greedy trajectory of a trained agent.
    """

    result = get_agent_trajectory(Q, seed=seed)

    fig, ax = plt.subplots(figsize=(8, 8))

    rows, cols = 6, 6

    # Draw grid cells.
    for r in range(rows):
        for c in range(cols):

            color = "white"

            if (r, c) in HAZARDS:
                color = "lightcoral"
            elif (r, c) == START:
                color = "lightgreen"
            elif (r, c) == GOAL:
                color = "gold"

            ax.add_patch(
                Rectangle(
                    (c - 0.5, r - 0.5),
                    1, 1,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1.5
                )
            )

    # Draw each continuous path segment.
    # Dashed lines indicate a reset after entering a hazard.
    first_segment = True

    for segment in result["segments"]:

        if len(segment) > 1:
            x = [state[1] for state in segment]
            y = [state[0] for state in segment]

            ax.plot(
                x, y,
                marker="o",
                linewidth=2.5,
                markersize=7,
                color="blue",
                label="Agent path" if first_segment else None
            )

            first_segment = False

        # Number the visited cells.
        for state in segment:
            r, c = state
            ax.text(
                c, r - 0.18,
                str(segment.index(state) + 1),
                ha="center",
                va="center",
                fontsize=8,
                color="black"
            )

    # Mark hazards.
    for r, c in HAZARDS:
        ax.text(
            c, r, "X",
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            color="darkred"
        )

    # Mark start and goal.
    ax.text(
        START[1], START[0] + 0.25,
        "START",
        ha="center",
        fontsize=8,
        fontweight="bold"
    )

    ax.text(
        GOAL[1], GOAL[0] + 0.25,
        "GOAL",
        ha="center",
        fontsize=8,
        fontweight="bold"
    )

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_aspect("equal")

    ax.set_title(
        f"{title}\n"
        f"Steps: {result['steps']} | "
        f"Hazards: {result['hazard_hits']} | "
        f"Success: {result['success']}"
    )

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.grid(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()

    # Create results folder if it doesn't exist
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Save plot
    filename = title.lower().replace(" ", "_").replace("-", "")
    save_path = results_dir / f"{filename}.png"

    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    print(f"Plot saved at: {save_path}")

    plt.show()
    plt.close()

    return result
