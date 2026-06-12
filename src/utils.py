"""Utility functions for logging and display."""
from __future__ import annotations

import logging
import os
from typing import Dict, Tuple

State = Tuple[int, int]


def setup_logger(log_path: str = "logs/sample_execution.log") -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
        force=True,
    )


def print_grid(grid):
    for row in grid:
        print("\t".join(f"{x:>7.2f}" if isinstance(x, float) else f"{x:>7}" for x in row))


def arrow(action: str) -> str:
    return {"right": "R", "down": "D", "left": "L", "up": "U"}.get(action, action)
