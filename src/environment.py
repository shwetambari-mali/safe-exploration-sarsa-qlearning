"""Grid-world environment for safe exploration experiments."""
from __future__ import annotations

import numpy as np

ROWS, COLS = 6, 6
START = (5, 0)
GOAL = (5, 5)

# Cliff cells along the bottom row, plus an additional hazard region.
HAZARDS = {
    (5, 1), (5, 2), (5, 3), (5, 4),
    (3, 2), (3, 3),
}

# Action IDs: up, down, left, right.
ACTIONS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}
N_ACTIONS = len(ACTIONS)
N_STATES = ROWS * COLS

STEP_REWARD = -1.0
HAZARD_REWARD = -20.0
GOAL_REWARD = 20.0


def state_to_id(state: tuple[int, int]) -> int:
    """Convert (row, column) into a flat state index."""
    row, col = state
    if not (0 <= row < ROWS and 0 <= col < COLS):
        raise ValueError(f"State is outside the grid: {state}")
    return row * COLS + col


def id_to_state(state_id: int) -> tuple[int, int]:
    """Convert a flat state index into (row, column)."""
    if not (0 <= state_id < N_STATES):
        raise ValueError(f"Invalid state ID: {state_id}")
    return divmod(state_id, COLS)


def step(state: tuple[int, int], action: int):
    """
    Apply one action.

    Ordinary move: -1 and continue.
    Hazard: -20, reset to START, and continue the episode.
    Goal: +20 and terminate the episode.
    Boundary: remain in the same cell and receive -1.
    """
    if action not in ACTIONS:
        raise ValueError(f"Invalid action: {action}")

    row, col = state
    dr, dc = ACTIONS[action]
    next_row, next_col = row + dr, col + dc

    if not (0 <= next_row < ROWS and 0 <= next_col < COLS):
        next_state = state
    else:
        next_state = (next_row, next_col)

    if next_state in HAZARDS:
        return START, HAZARD_REWARD, False

    if next_state == GOAL:
        return next_state, GOAL_REWARD, True

    return next_state, STEP_REWARD, False



def render_grid(ax=None):
    """Draw the grid world using custom colors."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Cell values:
    # 0 = Normal cell
    # 1 = Hazard
    # 2 = Start
    # 3 = Goal

    image = np.zeros((ROWS, COLS))

    for r, c in HAZARDS:
        image[r, c] = 1

    image[START] = 2
    image[GOAL] = 3

    # Custom color scheme
    colors = [
        "white",       # Normal cells
        "orange",      # Hazards
        "lightgreen",  # Start
        "green"        # Goal
    ]

    cmap = ListedColormap(colors)

    ax.imshow(image, cmap=cmap, vmin=0, vmax=3)

    # Add labels
    for r in range(ROWS):
        for c in range(COLS):

            if (r, c) == START:
                label = "START"
            elif (r, c) == GOAL:
                label = "GOAL"
            elif (r, c) in HAZARDS:
                label = "HAZARD"
            else:
                label = ""

            ax.text(
                c, r, label,
                ha="center",
                va="center",
                fontsize=7
            )

    # Grid lines
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))

    ax.set_xticks(np.arange(-0.5, COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, ROWS, 1), minor=True)

    ax.grid(which="minor", color="black", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title("Safe Exploration Grid World")

    return ax


