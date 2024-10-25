
# Project: Reinforcement Learning with Q-learning and SARSA

This project implements reinforcement learning algorithms to solve a grid-based navigation problem. Two learning methods, **Q-learning** and **SARSA**, are explored to train an agent to find the optimal path in an environment with potential traps and varying rewards.

## File Structure

- **`qlearning.ipynb`**: Notebook containing the Q-learning algorithm implementation, which uses an off-policy value update approach to learn the optimal policy.
- **`SARSA.ipynb`**: Notebook containing the SARSA algorithm, an on-policy reinforcement learning approach. SARSA updates the state-action values following the agent's current policy.
- **`utils.py`**: Utility functions for grid management, states, rewards, and policy visualization. This includes functions for visualizing value functions, transitions, and optimal policies.

## Key Features

### 1. Grid and Transition Generation
   - `dict_transition`: Generates possible transitions for each state based on grid dimensions and traps.
   - `dict_rewards`: Generates rewards associated with each position on the grid, including penalties for traps.

### 2. Q-learning (`qlearning.ipynb`)
   - State and action initialization.
   - Action selection using the epsilon-greedy method.
   - Q-function update according to the Q-learning formula.
   - Displays results and visualizes the optimal policy on the grid.

### 3. SARSA (`SARSA.ipynb`)
   - Implementation of the SARSA algorithm to update state-action pairs while following the agent's current policy.
   - Action selection using an epsilon-greedy policy.
   - Visualization of value function convergence for each state.

### 4. Utility Functions and Visualization (`utils.py`)
   - **Grid and Transitions**: Grid generation, trap management, and reward settings.
   - **Value Functions**: Value function initialization and update for each state.
   - **Action Selection**: Functions to select actions based on the policy.
   - **Visualization**: Displays value functions as heatmaps, visualizes transitions, and optimal policies.

### 5. Solution Graphs
The project provides graphical representations to compare the policies and results of each algorithm:
   - **Value Function Heatmap**: A heatmap to visualize the value function for each state.
   - **Optimal Path**: Visualizes the path taken by the agent following the optimal policy.
   - **Transition and Reward Maps**: Detailed grid maps for transitions and rewards, showcasing the influence of traps and terminal states.

## Installation and Requirements

1. **Python** 3.x
2. Required libraries (installable via pip):
   ```bash
   pip install numpy matplotlib seaborn
   ```

## Usage

1. **Configure Grid and Parameters**
   - Set grid dimensions (`K`), the terminal state position (`T`), and traps if necessary.
   
2. **Run Q-learning**
   - Open `qlearning.ipynb` and execute each cell to start training the agent.

3. **Run SARSA**
   - Open `SARSA.ipynb` and execute each cell to train the agent using the SARSA algorithm.

## Results

The project generates visualizations of the value functions for each algorithm, the optimal policies, and the paths taken by the agent. These results allow for a performance comparison between Q-learning and SARSA.

