# CSCN8020 Assignment 1 - Reinforcement Learning Programming

**Student:** Abdalla Mohamed  
**Student ID:** 9089339  
**Course:** CSCN8020 Reinforcement Learning Programming  

## Summary
This repository contains one complete Jupyter Notebook for Assignment 1. The work covers MDP design for a pick-and-place robot, manual and coded value iteration for a 2x2 gridworld, standard and in-place value iteration for a 5x5 gridworld, and off-policy Monte Carlo with weighted importance sampling. The implementation uses object-oriented Python classes for environments, agents, policies, and utilities. It also includes logging so the evaluator can verify the algorithm execution process.

## Repository Structure
```text
CSCN8020_Assignment1/
├── README.md
├── requirements.txt
├── .gitignore
├── CSCN8020_Assignment1.ipynb
├── src/
│   ├── environments.py
│   ├── agents.py
│   ├── policies.py
│   └── utils.py
├── images/
└── logs/
    └── sample_execution.log
```

## How to Run
```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/CSCN8020_Assignment1.git
cd CSCN8020_Assignment1
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook CSCN8020_Assignment1.ipynb
```

Run the notebook from top to bottom. A log file is generated at `logs/sample_execution.log`.

## Assumptions
- Discount factor: gamma = 0.9 unless otherwise stated.
- For the 5x5 gridworld, the regular-state reward is assumed to be -1 because the uploaded assignment table visually omits the regular reward value.
- State rewards in the 2x2 manual calculation are treated as rewards for being in the current state, as described in the assignment statement.

## References
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Course examples from CSCN8020 playground Lecture 2 MDP, Lecture 3 Dynamic Programming, and Lecture 4 Monte Carlo.
- Gymnasium documentation and course HelloGymMaze reference.
