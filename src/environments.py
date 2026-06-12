"""Gridworld environments for CSCN8020 Assignment 1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # keeps notebook readable if gymnasium is not installed yet
    gym = None
    spaces = None

State = Tuple[int, int]
Action = str


@dataclass(frozen=True)
class StepResult:
    next_state: State
    reward: float
    done: bool


class GridWorldEnvironment(gym.Env if gym else object):
    """Deterministic rectangular gridworld with wall handling and state-based rewards."""

    metadata = {"render_modes": ["ansi"]}
    ACTIONS: Tuple[Action, ...] = ("right", "down", "left", "up")
    DELTAS: Dict[Action, State] = {
        "right": (0, 1),
        "down": (1, 0),
        "left": (0, -1),
        "up": (-1, 0),
    }

    def __init__(
        self,
        rows: int,
        cols: int,
        terminal_states: Iterable[State] | None = None,
        grey_states: Iterable[State] | None = None,
        reward_goal: float = 10.0,
        reward_grey: float = -5.0,
        reward_regular: float = -1.0,
        max_steps: int = 100,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.states: List[State] = [(r, c) for r in range(rows) for c in range(cols)]
        self.terminal_states = set(terminal_states or [])
        self.grey_states = set(grey_states or [])
        self.reward_goal = float(reward_goal)
        self.reward_grey = float(reward_grey)
        self.reward_regular = float(reward_regular)
        self.max_steps = max_steps
        self.current_state: State = (0, 0)
        self.steps = 0
        if spaces is not None:
            self.action_space = spaces.Discrete(len(self.ACTIONS))
            self.observation_space = spaces.MultiDiscrete([rows, cols])

    def reward(self, state: State) -> float:
        if state in self.terminal_states:
            return self.reward_goal
        if state in self.grey_states:
            return self.reward_grey
        return self.reward_regular

    def is_terminal(self, state: State) -> bool:
        return state in self.terminal_states

    def valid_state(self, state: State) -> bool:
        r, c = state
        return 0 <= r < self.rows and 0 <= c < self.cols

    def transition(self, state: State, action: Action) -> StepResult:
        if self.is_terminal(state):
            return StepResult(state, 0.0, True)
        dr, dc = self.DELTAS[action]
        candidate = (state[0] + dr, state[1] + dc)
        next_state = candidate if self.valid_state(candidate) else state
        return StepResult(next_state, self.reward(next_state), self.is_terminal(next_state))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if gym is not None:
            super().reset(seed=seed)
        self.current_state = (0, 0)
        self.steps = 0
        return self.current_state, {}

    def step(self, action_index: int):
        action = self.ACTIONS[action_index]
        result = self.transition(self.current_state, action)
        self.current_state = result.next_state
        self.steps += 1
        truncated = self.steps >= self.max_steps and not result.done
        return result.next_state, result.reward, result.done, truncated, {}

    def values_as_grid(self, values: Dict[State, float]) -> List[List[float]]:
        return [[values[(r, c)] for c in range(self.cols)] for r in range(self.rows)]

    def policy_as_grid(self, policy: Dict[State, Action]) -> List[List[str]]:
        return [["G" if (r, c) in self.terminal_states else policy.get((r, c), "") for c in range(self.cols)] for r in range(self.rows)]


def make_assignment_gridworld() -> GridWorldEnvironment:
    """5x5 gridworld from the assignment. Regular-state reward is assumed to be -1."""
    return GridWorldEnvironment(
        rows=5,
        cols=5,
        terminal_states={(4, 4)},
        grey_states={(2, 2), (3, 0), (0, 4)},
        reward_goal=10,
        reward_grey=-5,
        reward_regular=-1,
        max_steps=100,
    )
