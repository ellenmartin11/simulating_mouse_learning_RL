import nbformat as nbf

#Initialize the notebook object
nb = nbf.v4.new_notebook()

#CONTENT

#Introduction
intro_text = """# Simulating Rodent Learning Curves with Reinforcement Learning Agents

## Project Overview
This project simulates mouse choice behavior in a non-stationary (switching) two-armed bandit task using various Reinforcement Learning (RL) agents.

* **Goal:** To fit a Reinforcement Learning (RL) model that closely mimics rodent learning and decision-making, retrieving best-fitting model parameters.
* **Data:** Beron et al. (2022).
* **Method:**
    * Agents are fit to the data using a **Randomized Search** to minimize the **Root Mean Squared Error (RMSE)** between the simulated learning curve and the observed mouse curve (averaged across 6 mice).
    * To correctly simulate the "re-learning"/confusion phase (after a block switch), agents are **pre-trained on the wrong arm** before the simulation begins (75 trials)
    * We analyze performance across three reward probability conditions: **70-30**, **80-20**, and **90-10**.

## Table of Contents
1.  [Model 1: Epsilon Greedy Q-Learning](#1.-Model-1:-Epsilon-Greedy-Q-Learning)
2.  [Model 2: Q-Learning with Forgetting/Satiation](#2.-Model-2:-Q-Learning-with-Forgetting)
3.  [Model 3: Q-Learning with Anticipation and Boltzmann Exploration)](#3.-Model-3:-Q-Learning-with-Anticipation-(Boltzmann))
4.  [Summary of Findings](#4.-Summary-of-Findings)
"""

#  MODEL 1: EPSILON GREEDY 
m1_intro = """# 1. Model 1: Epsilon Greedy Q-Learning

### Theory
This is the baseline model. It assumes mice learn value estimates ($Q$) and explore randomly with probability $\epsilon$ (coin-toss if randomly generated value < $\epsilon$). Exploration decays over time to simulate increasing exploitation of a known good policy.

**Q-Update Rule:**
$$Q(a_{t+1}) = Q(a_t) + \\alpha \\cdot [reward - Q(a_t)]$$

**Action Selection:**
* With probability $\\epsilon$: Choose random action.
* With probability $1-\\epsilon$: Choose $argmax Q(a_t)$.

**Decay:**
$$\\epsilon_{t+1} = \\max(\\epsilon_{min}, \\epsilon_t \\cdot decayRate)$$
"""

m1_70_30 = """## Condition 70-30 (More Uncertainty)

### Fits and Q-Values
| Full Analysis (All Trials) | Truncated Analysis (First 100 Trials) |
| :---: | :---: |
| ![Full](plots/plot_epsilon_greedy_70-30_fit_RMSE.png) | ![Truncated](plots/plot_epsilon_greedy_70-30_fit_RMSE_T100.png) |

![Q-Values](plots/plot_epsilon_greedy_70-30_Q_Values.png)

### Analysis
* **RMSE (Full):** 0.0776
* **RMSE (T=100):** 0.028 (Excellent fit)
* **Parameters:** $\\alpha \\approx 0.41$, $\\epsilon_{min} \\approx 0.14$
* **Observation:** The model captures the general learning of real mice well, but the simulated agent learns slightly faster than the average mouse. The Q-value plot shows the agent accurately estimates the value of the 'good' arm, though may slightly underestimate.
"""

m1_80_20 = """## Condition 80-20 (Medium Uncertainty)

### Fits and Q-Values
| Full Analysis (All Trials) | Truncated Analysis (First 100 Trials) |
| :---: | :---: |
| ![Full](plots/plot_epsilon_greedy_80-20_fit_RMSE.png) | ![Truncated](plots/plot_epsilon_greedy_80-20_fit_RMSE_T100.png) |

![Q-Values](plots/plot_epsilon_greedy_80-20_Q_Values.png)

### Analysis
* **RMSE (Full):** 0.069
* **RMSE (T=100):** 0.024
* **Parameters:** $\\alpha \\approx 0.53$, $\\epsilon_{min} \\approx 0.09$
* **Observation:** Performance is slightly better than the 70-30 condition. The agent exploits more (lower $\\epsilon_{min}$), matching the easier task - it is more obvious that one arm is better than the other. However, the simulation peaks at >90% accuracy, whereas real mice remain more stochastic.
"""

m1_90_10 = """## Condition 90-10 (Low Uncertainty)

### Fits and Q-Values
| Full Analysis (All Trials) | Truncated Analysis (First 100 Trials) |
| :---: | :---: |
| ![Full](plots/plot_epsilon_greedy_90-10_fit_RMSE.png) | ![Truncated](plots/plot_epsilon_greedy_90-10_fit_RMSE_T100.png) |

![Q-Values](plots/plot_epsilon_greedy_90-10_Q_Values.png)

### Analysis
* **RMSE (Full):** 0.139 (Very poor/unacceptable fit)
* **RMSE (T=100):** 0.045 (Good fit)
* **Parameters:** $\\alpha \\approx 0.50$, $\\epsilon_{min} \\approx 0.11$
* **Observation:** This is the worst fit for the baseline model. Real mice are surprisingly stochastic in this easy condition, whereas the optimal agent quickly converges to almost perfect performance. The rightmost tail of the mouse data drops off, which this model cannot capture. This is most likely a statistical artefact of the study design, where the average length of block trials was 50, according to Beron et al (2022), so there are few (noisy) observations of mouse behaviour in these latter trials. 
"""

# MODEL 2: FORGETTING 
m2_intro = """# 2. Model 2: Q-Learning with Forgetting

### Theory
To address the decline in performance seen in later trials (especially in 90-10), we add a **Global Forgetting / Satiation** parameter $\\omega$. This scales down *all* Q-values on every trial, simulating either 'leaky' memory or a loss of motivation (satiation). This makes sense considering the study design which was to deprive mice of water prior to trials. 

**Update Rule:**
$$Q(a_{all}) = Q(a_{all}) \\cdot \\omega$$
$$Q(a_{chosen}) = Q(a_{chosen}) + \\alpha \\cdot [reward - Q(a_{chosen})]$$
"""

m2_70_30 = """## Condition 70-30

### Fits and Q-Values
| Full Analysis | Truncated Analysis (T=100) |
| :---: | :---: |
| ![Full](plots/plot_forgetting_70-30_fit_RMSE.png) | ![Truncated](plots/plot_forgetting_70-30_fit_RMSE_T100.png) |

![Q-Values](plots/plot_forgetting_70-30_Q_Values.png)

### Analysis
* **RMSE (Full):** 0.103 (Worse than baseline)
* **RMSE (T=100):** 0.07
* **Observation:** The forgetting parameter leads to a consistent underestimation of Q-values, which is unsurprising. It does not successfully capture the inverted U shape of the mouse behavior, it just lowers the overall learning curve. Again, the model based on truncated data performs better, because there is less noise. 
"""

m2_80_20 = """## Condition 80-20

### Fits and Q-Values
| Full Analysis | Truncated Analysis (T=100) |
| :---: | :---: |
| ![Full](plots/plot_forgetting_80-20_fit_RMSE.png) | ![Truncated](plots/plot_forgetting_80-20_fit_RMSE_T100.png) |

![Q-Values](plots/plot_forgetting_80-20_Q_Values.png)

### Analysis
* **RMSE (Full):** 0.093
* **RMSE (T=100):** 0.024
* **Observation:** The model fits the early learning phase well (T=100) but fails to improve upon the baseline model (Epsilon-Greedy). The best-fit parameters seem to select  very low forgetting ($\omega \\approx 0.999$), suggesting that 'forgetting' doesn't add much to the baseline model.
"""

m2_90_10 = """## Condition 90-10

### Fits and Q-Values
| Full Analysis | Truncated Analysis (T=100) |
| :---: | :---: |
| ![Full](plots/plot_forgetting_90-10_fit_RMSE.png) | ![Truncated](plots/plot_forgetting_90-10_fit_RMSE_T100.png) |

![Q-Values](plots/plot_forgetting_90-10_Q_Values.png)

### Analysis
* **RMSE (Full):** 0.149
* **RMSE (T=100):** 0.061
* **Observation:** The high rewards in this condition make the forgetting factor more damaging for model performance. Q-values are consistently underestimated, suggesting that simple "global, passive decay" is not the mechanism behind the mouse's late-block stochasticity. Again, this is likely to be noise, as the model does quite well on the first 100 trials. 
"""

#  MODEL 3: ANTICIPATION 
m3_intro = """# 3. Model 3: Q-Learning with Anticipation (Boltzmann Exploration)

### Theory
Given the repeated trials design, mice may learn the *structure* of the task, and eventually learn that a switch in rewarding arms is coming at some point. This model includes **Anticipation**.
Instead of $\\epsilon$-greedy, we use **Boltzmann Exploration** (Softmax), controlled by a dynamic Temperature ($\\tau$) parameter.

1.  **Learning Phase:** $\\tau$ decays (cools down), encouraging exploitation of policy.
2.  **Anticipation Phase:** After a certain trial threshold, $\\tau$ increases (heats up), encouraging exploration in anticipation of a switch!

**Softmax Probability:**
$$P(a_i) = \\frac{e^{Q(a_i)/\\tau}}{e^{Q(a_1)/\\tau} + e^{Q(a_2)/\\tau}}$$

**Temperature:**
If $t > anticipationTrial$: $\\tau_{t+1} = \\tau_t \\cdot anticipationRate$
"""

m3_70_30 = """## Condition 70-30

### Fits and Q-Values
| Full Analysis | Truncated Analysis (T=100) |
| :---: | :---: |
| ![Full](plots/plot_boltzmann_anticipation_70-30_fit_RMSE.png) | ![Truncated](plots/plot_boltzmann_anticipation_70-30_fit_RMSE_T100.png) |

![Q-Values](plots/plot_boltzmann_anticipation_70-30_Q_Values.png)

### Analysis
* **RMSE (Full):** 0.096
* **RMSE (T=100):** 0.040
* **Observation:** This model begins to capture the "tail" stochasticity better when fit to the full data. We see the simulated curve dip slightly in later trials, mirroring the mouse's increased stochasticity. RMSE is good, but notably better when fit on just the earlier trials. 
"""

m3_80_20 = """## Condition 80-20

### Fits and Q-Values
| Full Analysis | Truncated Analysis (T=100) |
| :---: | :---: |
| ![Full](plots/plot_boltzmann_anticipation_80-20_fit_RMSE.png) | ![Truncated](plots/plot_boltzmann_anticipation_80-20_fit_RMSE_T100.png) |

![Q-Values](plots/plot_boltzmann_anticipation_80-20_Q_Values.png)

### Analysis
* **RMSE (Full):** 0.083
* **RMSE (T=100):** 0.031
* **Observation:** The Q-values converge quickly (25 trials). The anticipation parameter models the variance later in trial blocks without hugely impacting early learning performance. 
"""

m3_90_10 = """## Condition 90-10

### Fits and Q-Values
| Full Analysis | Truncated Analysis (T=100) |
| :---: | :---: |
| ![Full](plots/plot_boltzmann_anticipation_90-10_fit_RMSE.png) | ![Truncated](plots/plot_boltzmann_anticipation_90-10_fit_RMSE_T100.png) |

![Q-Values](plots/plot_boltzmann_anticipation_90-10_Q_Values.png)

### Analysis
* **RMSE (Full):** 0.140
* **Observation:** This model struggles with the 90-10 condition. The mice are surprisingly suboptimal here compared to the agent, which is unexpected considering it is quite obvious which arm is the more rewarding.
"""

# SUMMARY
summary_text = """# 4. Summary of Findings

1.  **Re-Learning vs. Fresh Learning:** Accurately simulating the mouse's behavior required "pre-training" agents on the wrong arm (for 75 trials). Suggests that the mouse behavior is a "re-learning" curve, and mice have to work to overcome previously learnt information. 
2.  **Baseline Performance:** The simple **Epsilon-Greedy** model provided a surprisingly good fit for the first 100 trials, particularly in the 70-30 and 80-20 conditions, performing poorly (like all other models) for the 90-10 condition, which is somewhat surprising given that this is the **least uncertain** of the three environmental conditions. 
3.  **Forgetting/Satiation Hypothesis:** The "Forgetting/Satiation" model failed to improve fits. The "inverted U" shape of the mouse curve is likely not a result of memory decay or satiation, but more likely a statistical artefact and product of few behavioural observations. 
4.  **Anticipation Hypothesis:** The **Boltzmann Anticipation** model provided the most possible explanation (apart from noise) for the stochasticity in late-trial behaviour. By increasing (exploration) in later trials, it captures the mouse's tendency to "check"/switch to the other arm from time to time, despite likely knowing what the best arm is, as the probability of a block switch increases (trial number increases). This was the best performing model for the 90-10 condition. 
"""

# Add cells to notebook
nb.cells.append(nbf.v4.new_markdown_cell(intro_text))
nb.cells.append(nbf.v4.new_markdown_cell(m1_intro))
nb.cells.append(nbf.v4.new_markdown_cell(m1_70_30))
nb.cells.append(nbf.v4.new_markdown_cell(m1_80_20))
nb.cells.append(nbf.v4.new_markdown_cell(m1_90_10))
nb.cells.append(nbf.v4.new_markdown_cell(m2_intro))
nb.cells.append(nbf.v4.new_markdown_cell(m2_70_30))
nb.cells.append(nbf.v4.new_markdown_cell(m2_80_20))
nb.cells.append(nbf.v4.new_markdown_cell(m2_90_10))
nb.cells.append(nbf.v4.new_markdown_cell(m3_intro))
nb.cells.append(nbf.v4.new_markdown_cell(m3_70_30))
nb.cells.append(nbf.v4.new_markdown_cell(m3_80_20))
nb.cells.append(nbf.v4.new_markdown_cell(m3_90_10))
nb.cells.append(nbf.v4.new_markdown_cell(summary_text))

# Write the notebook
with open('Formatted_Results.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook 'Formatted_Results.ipynb' created successfully!")