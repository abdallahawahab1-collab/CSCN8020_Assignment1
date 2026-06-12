"""Policy helpers."""
from __future__ import annotations

import random
from typing import Dict, Sequence, Tuple

State = Tuple[int, int]
Action = str


class RandomPolicy:
    def __init__(self, actions: Sequence[Action], seed: int = 42) -> None:
        self.actions = list(actions)
        self.random = random.Random(seed)

    def action(self, state: State) -> Action:
        return self.random.choice(self.actions)

    def probability(self, state: State, action: Action) -> float:
        return 1.0 / len(self.actions)


class GreedyPolicy:
    def __init__(self, action_by_state: Dict[State, Action], actions: Sequence[Action]) -> None:
        self.action_by_state = action_by_state
        self.actions = list(actions)

    def action(self, state: State) -> Action:
        return self.action_by_state.get(state, self.actions[0])

    def probability(self, state: State, action: Action) -> float:
        return 1.0 if self.action(state) == action else 0.0
