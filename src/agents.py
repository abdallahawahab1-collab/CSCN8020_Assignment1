"""Dynamic programming and Monte Carlo agents."""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from typing import Dict, List, Tuple

from .environments import GridWorldEnvironment, State, Action
from .policies import RandomPolicy


class ValueIterationAgent:
    def __init__(self, env: GridWorldEnvironment, gamma: float = 0.9, theta: float = 1e-8, max_iterations: int = 10_000):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.values: Dict[State, float] = {s: 0.0 for s in env.states}
        self.policy: Dict[State, Action] = {}
        self.iterations = 0
        self.optimization_time = 0.0

    def q_value(self, state: State, action: Action, values: Dict[State, float] | None = None) -> float:
        values = self.values if values is None else values
        step = self.env.transition(state, action)
        return step.reward + self.gamma * values[step.next_state]

    def run(self) -> Tuple[Dict[State, float], Dict[State, Action]]:
        logging.info("Starting synchronous value iteration: gamma=%s theta=%s", self.gamma, self.theta)
        start = time.perf_counter()
        for iteration in range(1, self.max_iterations + 1):
            delta = 0.0
            new_values = self.values.copy()
            for state in self.env.states:
                if self.env.is_terminal(state):
                    new_values[state] = self.env.reward(state)
                    continue
                best = max(self.q_value(state, a, self.values) for a in self.env.ACTIONS)
                delta = max(delta, abs(best - self.values[state]))
                new_values[state] = best
            self.values = new_values
            self.iterations = iteration
            logging.info("VI iteration=%s delta=%.10f", iteration, delta)
            if delta < self.theta:
                break
        self.optimization_time = time.perf_counter() - start
        self.policy = self.extract_policy()
        logging.info("Finished VI iterations=%s time=%.6f", self.iterations, self.optimization_time)
        return self.values, self.policy

    def extract_policy(self) -> Dict[State, Action]:
        policy: Dict[State, Action] = {}
        for state in self.env.states:
            if self.env.is_terminal(state):
                continue
            scores = {a: self.q_value(state, a) for a in self.env.ACTIONS}
            policy[state] = max(scores, key=scores.get)
        return policy


class InPlaceValueIterationAgent(ValueIterationAgent):
    def run(self) -> Tuple[Dict[State, float], Dict[State, Action]]:
        logging.info("Starting in-place value iteration: gamma=%s theta=%s", self.gamma, self.theta)
        start = time.perf_counter()
        for iteration in range(1, self.max_iterations + 1):
            delta = 0.0
            for state in self.env.states:
                if self.env.is_terminal(state):
                    old = self.values[state]
                    self.values[state] = self.env.reward(state)
                    delta = max(delta, abs(old - self.values[state]))
                    continue
                old = self.values[state]
                self.values[state] = max(self.q_value(state, a, self.values) for a in self.env.ACTIONS)
                delta = max(delta, abs(old - self.values[state]))
            self.iterations = iteration
            logging.info("In-place VI iteration=%s delta=%.10f", iteration, delta)
            if delta < self.theta:
                break
        self.optimization_time = time.perf_counter() - start
        self.policy = self.extract_policy()
        logging.info("Finished in-place VI iterations=%s time=%.6f", self.iterations, self.optimization_time)
        return self.values, self.policy


class OffPolicyMonteCarloAgent:
    """Off-policy MC control/prediction with weighted importance sampling."""

    def __init__(self, env: GridWorldEnvironment, gamma: float = 0.9, episodes: int = 20_000, seed: int = 42):
        self.env = env
        self.gamma = gamma
        self.episodes = episodes
        self.behavior = RandomPolicy(env.ACTIONS, seed=seed)
        self.q: Dict[Tuple[State, Action], float] = defaultdict(float)
        self.c: Dict[Tuple[State, Action], float] = defaultdict(float)
        self.policy: Dict[State, Action] = {s: env.ACTIONS[0] for s in env.states if not env.is_terminal(s)}
        self.values: Dict[State, float] = {s: 0.0 for s in env.states}
        self.optimization_time = 0.0

    def greedy_action(self, state: State) -> Action:
        scores = {a: self.q[(state, a)] for a in self.env.ACTIONS}
        return max(scores, key=scores.get)

    def generate_episode(self) -> List[Tuple[State, Action, float]]:
        state, _ = self.env.reset()
        episode: List[Tuple[State, Action, float]] = []
        for _ in range(self.env.max_steps):
            action = self.behavior.action(state)
            step = self.env.transition(state, action)
            episode.append((state, action, step.reward))
            state = step.next_state
            if step.done:
                break
        return episode

    def run(self) -> Tuple[Dict[State, float], Dict[State, Action]]:
        logging.info("Starting off-policy MC: gamma=%s episodes=%s", self.gamma, self.episodes)
        start = time.perf_counter()
        for episode_num in range(1, self.episodes + 1):
            episode = self.generate_episode()
            G = 0.0
            W = 1.0
            for state, action, reward in reversed(episode):
                G = self.gamma * G + reward
                key = (state, action)
                self.c[key] += W
                self.q[key] += (W / self.c[key]) * (G - self.q[key])
                self.policy[state] = self.greedy_action(state)
                if action != self.policy[state]:
                    break
                W *= 1.0 / self.behavior.probability(state, action)
            if episode_num % max(1, self.episodes // 10) == 0:
                logging.info("MC episode=%s/%s", episode_num, self.episodes)
        for state in self.env.states:
            if self.env.is_terminal(state):
                self.values[state] = self.env.reward(state)
            else:
                self.values[state] = max(self.q[(state, a)] for a in self.env.ACTIONS)
        self.optimization_time = time.perf_counter() - start
        logging.info("Finished MC episodes=%s time=%.6f", self.episodes, self.optimization_time)
        return self.values, self.policy
