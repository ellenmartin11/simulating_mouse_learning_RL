# Simulating Mouse Learning Strategies with Reinforcement Learning Models

This repository contains the final project for DSCI 6612 - Intro to AI at the University of New Haven. The goal of this project is to model and simulate mouse behavior in a dynamic two-armed bandit task, with multiple reward probability conditions.

This project implements several reinforcement learning (RL) agents with increasing complexity to simulate mouse learning and decision-making, from a simple Epsilon-Greedy Q-learner to a complex Boltzmann agent with anticipatory switching behaviour (picking the non-rewarding arm). These agents are fit to real behavioral data from Beron et al (2022) to find the model and parameters that best explain the average learning of mice in the two-armed bandit task. 

**Author:** \[Ellen Martin\]
**Date:** \[November 2025\]

## 1. Project Objective

The core question of this project is: **How might an average mouse be learning about the rewards in a dynamic two-armed bandit task?**

A simple "optimal" RL agent will learn the best choice (quickly) and exploit this to achieve the greatest expected utility. Real-world mouse data tends to show more stochastic, unpredictable and, by definition, suboptimal behaviour. Accordingly, this project aims to find the best-fitting agent that can *mimic* this complex, sub-optimal, though psychologically realistic behavior.

This project does not aim to replicate the methods and findings of Beron et al (2022), but instead to use their real-life mouse dataset as a foundation for exploring methods in Reinforcement Learning to simulate organic, stochastic decision-making. The goal is also not to generate an optimal agent. 

## 2. Data Source

The behavioral data used is from the Beron et al. (2022) study on mouse decision-making. The raw data (`bandit_data.csv`) contains trial-by-trial choices, rewards, and task conditions for mice engaged in a two-armed bandit task.

* **Beron, C., et al. (2022).** *Mice exhibit stochastic and efficient action switching during probabilistic decision making.* PNAS.

* **Data Repository:** \[[Link to Beron et al. GitHub](https://github.com/celiaberon/2ABT_behavior_models)\]

In this task, the rewarding arm switches periodically, forcing mice to re-learn which arm is the rewarding one. Mice participate in three different reward conditions: 

1) 90%-10%: Rewarding arm dispenses a reward (water) 90% of the time, unrewarding arm dispenses a reward 10% of the time.
2) 80%-20%: Rewarding arm rewards 80% of the time, unrewarding arm rewards 20% of the time. 
3) 70%-30%: Rewarding arm rewards 70% of the time, unrewarding arm rewards 30% of the time. 

Because mice either receive a reward, or not, the expected value (used in Reinforcement Learning algorithms) is just 1*reward probability (0.9, 0.8 or 0.7), or zero (0, 0, 0). Fully optimal q-learning agents should therefore estimate q-values that eventually approach the actual reward probabilities (i.e., q = 0.9, 0.8, 0.7). 

As this project will demonstrate, not all Reinforcement Learning model specifications lead to accurate q-value estimations. 



## 3. How to Run This Project

### Dependencies

This project requires Python and the following libraries. You can install them via pip:

```bash
pip install pandas numpy matplotlib
```

### Main Script: `fitRLmodel_RMSE.py`

This is the main script used for all analysis. It is run from the command line and allows you to fit any defined agent to any of the three reward conditions.

For simpler models, 1000 simulations can be specified, with 50 search parameter sets to try.

For the more complex Boltzmann Anticipation agent, I recommend 500 simulations, reducing to 30 parameter sets. 

Running the terminal commands will generate a plot comparing the specified agent's average behaviour (over specified number of simulations) with the average mouse behaviour (averaged over all mice in Beron et al (2022)). A second plot will also be generated, illustrating the agent's estimate of Q-values across trials.

**Usage:**

```bash
python fitRLmodel_RMSE_T.py -m [MODEL_NAME] -c [CONDITION] --sims_per_set [NUMBER OF SIMULATIONS] --search_iterations [PARAM SEARCH ITER] --max_trials [TRIALS TO SIMULATE LEARNING]
```

**Arguments:**

* `-m, --model`: (Required) The name of the agent model to run.

  * Choices: `epsilon_greedy`, `epsilon_greedy_perserv`, `forgetting`, `boltzmann_anticipation`

* `-c, --condition`: (Required) The reward probability condition to test.

  * Choices: `70` (for 70-30), `80` (for 80-20), `90` (for 90-10)

* `--datafile`: (Optional) Path to data. (Default: `bandit_data.csv`)

* `--search_iterations`: (Optional) How many random parameter sets to try. (Default: `50`)

* `--sims_per_set`: (Optional) How many agents to simulate *per* parameter set. (Default: `100`)

* `--max_trials`: (Optional) How many trials to simulate. Default = n_trials

### Example Commands

**To fit the `boltzmann_anticipation` model to the 80-20 data:**

``` bash
python fitRLmodel_RMSE.py -m boltzmann_anticipation -c 80 --search_iterations 30 --sims_per_set 500
```

**To fit the simpler `epsilon_greedy` model to the 70-30 data:**

``` bash
python fitRLmodel_RMSE.py -m boltzmann_anticipation -c 80 --search_iterations 50 --sims_per_set 1000
```

### Quick View of Results

Key results and model interpretations can be found in the [Results Jupyter Notebook](results.ipynb). This notebook includes main models implemented on all three reward conditions, parameter estimates from simulations, simulated agent learning compared to average mouse learning (RMSE), as well as learning curves and q-value history curves. 

