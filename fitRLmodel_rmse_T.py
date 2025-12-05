import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import warnings
import argparse
from itertools import product
import time
import sys
import os

#HELPER FUNCTIONS (calculating RMSE, parsing experimental conditions)
def calculate_rmse(agent_curve, mouse_curve):
    """Calculates the RMSE between two average behavior curves."""
    min_length = min(len(agent_curve), len(mouse_curve))
    agent_data = agent_curve[:min_length]
    mouse_data = mouse_curve[:min_length]
    return np.sqrt(np.mean((agent_data - mouse_data)**2))

def parse_condition_arg(c):
    """Get specific probability condition from string in 'Condition' column of dataset."""
    if c == '70':
        condition_str = '70-30'
        prob_high, prob_low = 0.7, 0.3
    elif c == '80':
        condition_str = '80-20'
        prob_high, prob_low = 0.8, 0.2
    elif c == '90':
        condition_str = '90-10'
        prob_high, prob_low = 0.9, 0.1
    else:
        raise ValueError(f"Invalid condition: {c}. Must be '70', '80', or '90'.")
    return condition_str, prob_high, prob_low


# REINFORCEMENT LEARNING AGENT SPECIFICATIONS
# BASIC EPSILON GREEDY Q LEARNING 
class QLearningAgent_EpsilonGreedy:
    """
    Q learning agent with only epsilon decay.
    """
    def __init__(self, learning_rate, min_epsilon, decay_rate, start_epsilon=1.0, num_actions=2):
        self.alpha = learning_rate
        self.epsilon = start_epsilon
        self.start_epsilon = start_epsilon #for resetting
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.num_actions = num_actions
        self.q_values = np.zeros(num_actions)

    def choose_action(self, trial_number):
        if random.uniform(0,1) < self.epsilon: 
            action = random.choice([0,1]) #COIN FLIP to decide action (in this case right or left)
        else:
            action = np.argmax(self.q_values) #otherwise q-maximizing action
            
        if self.epsilon > self.min_epsilon: #apply decay to epsilon to avoid continuous exploration
            self.epsilon *= self.decay_rate
        return action
    
    def update_q_value(self, action, reward, trial_number): #q(a) <- q(a) + alpha[r - q(a)]
        self.q_values[action] = self.q_values[action] + self.alpha * (reward - self.q_values[action])
        
    def reset(self): #reset epsilon and q-values for each simulation
        self.epsilon = self.start_epsilon
        self.q_values = np.zeros(self.num_actions)
        
# Q LEARNING AGENT WITH NO EPSILON DECAY
class QLearningAgent_Epsilon:
    """ Q Learning agent with Epsilon greedy explore-exploit behaviour, but NO decay of epsilon over time."""
    
    def __init__(self, learning_rate, epsilon, num_actions=2):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.num_actions = num_actions
        
    def choose_action(self, trial_number):
        if random.uniform(0,1) < self.epsilon:
            action = random.choice([0,1])
        else:
            action = np.argmax(self.q_values)
        return action
    
    def update_q_value(self, action, reward, trial_number):
        self.q_values[action] = self.q_values[action] + self.learning_rate * (reward - self.q_values[action])
        
    def reset(self):
        self.q_values = np.zeros(self.num_actions)
        
        
# Q LEARNING AGENT WITH EPSILON GREEDY DECAY AND PERSEVERANCE
class QLearningAgent_EpsilonGreedyPersev:
    """
    Q learning agent with epsilon decay AND perseverance bias.
    """
    def __init__(self, learning_rate, min_epsilon, decay_rate, perseverance_bonus, start_epsilon=1.0, num_actions=2):
        self.alpha = learning_rate
        self.epsilon = start_epsilon
        self.start_epsilon = start_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.perseverance_bonus = perseverance_bonus #perseverance agent - might not report because it doesn't really add anything...
        self.last_action = None
        self.num_actions = num_actions
        self.q_values = np.zeros(num_actions)

    def choose_action(self, trial_number):
        if random.uniform(0,1) < self.epsilon: #same code as before
            action = random.choice([0,1])
        else:
            effective_q_values = self.q_values.copy() #make a copy of q-values
            if self.last_action is not None: #if previously an action was taken
                effective_q_values[self.last_action] += self.perseverance_bonus #give a little bonus for taking the same action again (perseverance/sticky bonus)
            action = np.argmax(effective_q_values) #same maximizing action selection though
            
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay_rate #decay
        self.last_action = action
        return action
    
    def update_q_value(self, action, reward, trial_number):
        self.q_values[action] = self.q_values[action] + self.alpha * (reward - self.q_values[action])
        if reward == 0 and action == self.last_action:
             self.last_action = None
        else:
             self.last_action = action

    def reset(self): #reset
        self.epsilon = self.start_epsilon
        self.q_values = np.zeros(self.num_actions)
        self.last_action = None

# Q LEARNING AGENT WITH ADDITIONAL'FORGETTING' 
class QLearningAgent_Forgetting:
    """
    Q learning agent with only epsilon decay and a forgetting factor.
    """
    def __init__(self, learning_rate, min_epsilon, decay_rate, forgetting, start_epsilon=1.0, num_actions=2):
        self.alpha = learning_rate
        self.epsilon = start_epsilon
        self.start_epsilon = start_epsilon 
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.forgetting = forgetting
        self.num_actions = num_actions
        self.q_values = np.zeros(num_actions)

    def choose_action(self, trial_number): 
        if random.uniform(0, 1) < self.epsilon: #same coin toss
            action = random.choice([0, 1])
        else:
            action = np.argmax(self.q_values)
            
        if self.epsilon > self.min_epsilon: #epsilon decay
            self.epsilon *= self.decay_rate
        return action

    def update_q_value(self, action, reward, trial_number): 
        self.q_values *= self.forgetting
        self.q_values[action] = self.q_values[action] + self.alpha * (reward - self.q_values[action])

    def reset(self):
        self.epsilon = self.start_epsilon
        self.q_values = np.zeros(self.num_actions)
        
# BOLTZMANN ANTICIPATION AGENT
class QLearningAgent_BoltzmannAnticipation:
    """
    Boltzmann (Softmax) Choice + Anticipation
    """
    def __init__(self, learning_rate, min_temperature, decay_rate, 
                 anticipation_trial, anticipation_rate, start_temperature=1.0, max_temperature=1.0, num_actions=2):
        self.alpha = learning_rate
        self.start_temperature = start_temperature
        self.tau = start_temperature
        self.min_temperature = min_temperature
        self.decay_rate = decay_rate
        self.last_action = None
        self.anticipation_trial = anticipation_trial
        self.anticipation_rate = anticipation_rate
        self.max_temperature = max_temperature
        self.num_actions = num_actions
        self.q_values = np.zeros(num_actions)
        self.is_learning_phase = True 

    def choose_action(self, trial_number):
        effective_q_values = self.q_values.copy()
            
        scaled_q = effective_q_values / (self.tau + 1e-7)
        exp_values = np.exp(scaled_q - np.max(scaled_q))
        probabilities = exp_values / np.sum(exp_values)
        action = np.random.choice(self.num_actions, p=probabilities)
        self.last_action = action
        return action

    def update_q_value(self, action, reward, trial_number):
        self.q_values[action] = self.q_values[action] + self.alpha * (reward - self.q_values[action])
        if reward == 0 and action == self.last_action:
             self.last_action = None
        else:
             self.last_action = action 
        if self.is_learning_phase and (trial_number >= self.anticipation_trial or self.tau <= self.min_temperature):
            self.is_learning_phase = False
        if self.is_learning_phase:
            if self.tau > self.min_temperature:
                self.tau *= self.decay_rate
        else:
            if self.tau < self.max_temperature:
                self.tau *= self.anticipation_rate
    
    def reset(self):
        self.tau = self.start_temperature
        self.last_action = None
        self.q_values = np.zeros(self.num_actions)
        self.is_learning_phase = True
    

#Model Registry & Search Ranges
MODELS = {
    'epsilon_greedy': {
        'class': QLearningAgent_EpsilonGreedy,
        'param_ranges': {
            'start_epsilon': (1.0, 1.0),
            'learning_rate': (0.01, 0.9),
            'min_epsilon': (0.01, 0.5),
            'decay_rate': (0.95, 0.999)
        }
    },
    'epsilon': {
        'class': QLearningAgent_Epsilon,
        'param_ranges': {
            'epsilon': (0.05, 0.95),
            'learning_rate': (0.01, 0.9)
            }
    },
    'epsilon_greedy_perserv': {
        'class': QLearningAgent_EpsilonGreedyPersev,
        'param_ranges': {
            'start_epsilon': (1.0, 1.0),
            'learning_rate': (0.01, 0.9),
            'min_epsilon': (0.01, 0.5),
            'decay_rate': (0.95, 0.999),
            'perseverance_bonus': (0.0, 0.5)
        }
    },
    'forgetting': {
        'class': QLearningAgent_Forgetting,
        'param_ranges': {
            'start_epsilon': (1.0, 1.0), 
            'learning_rate': (0.1, 0.4),
            'min_epsilon': (0.01, 0.01), 
            'decay_rate': (0.8, 0.99),
            'forgetting': (0.9, 0.999) 
        }
    },
    'boltzmann_anticipation': {
        'class': QLearningAgent_BoltzmannAnticipation,
        'param_ranges': {
            'start_temperature': (0.5, 2.0),
            'learning_rate': (0.01, 0.9),
            'min_temperature': (0.01, 0.5),
            'decay_rate': (0.95, 0.999),
            'anticipation_trial': (100, 250), # Integer
            'anticipation_rate': (1.001, 1.01),
            'max_temperature': (0.5, 2.0)
        }
    }
}

#Core Functions

def load_and_prep_data(datafile, condition_str, max_trials=None): 
    """Loads and prepares data, and returns the average mouse curve."""
    try:
        data = pd.read_csv(datafile)
    except FileNotFoundError:
        print(f"Error: Data file '{datafile}' not found.")
        sys.exit()
    
    data_condition = data[data['Condition'] == condition_str].copy()
    if data_condition.empty:
        print(f"Error: No data found for condition '{condition_str}'.")
        sys.exit()

    #Filter by max_trials before doing any analysis
    if max_trials is not None:
        print(f"Truncating analysis to trials 1 through {max_trials}.")
        data_condition = data_condition[data_condition['blockTrial'] <= max_trials].copy()

    data_condition['correct_choice'] = (data_condition['Decision'] == data_condition['Target']).astype(int)
    average_mouse_behavior = data_condition.groupby('blockTrial')['correct_choice'].mean()
    mouse_curve_data = average_mouse_behavior.values
    
    print(f"Data loaded for {condition_str}. Analyzing {len(mouse_curve_data)} trials.")
    return data_condition, mouse_curve_data

def simulate_agent_with_params(model_name, params_dict, num_trials, prob_high, prob_low, num_simulations):
    """
    Runs N simulations using a GIVEN set of parameters.
    This simulates the re-learning of which the valuable arm is, given random switches in the study design. 
    Otherwise the model fit will be atrocious, and mice will appear much more stochastic and suboptimal compared to the agents, when actually they are behaving well given the stocahstic environment.
    """
    model_class = MODELS[model_name]['class']
    agent_behavior = np.zeros((num_simulations, num_trials))
    
    #STORE Q-VALUES TO EXAMINE ESTIMATES
    #NUMBER OF ACTIONS
    _temp_agent = model_class(**params_dict) #temporary agent inherets everything else
    num_actions = _temp_agent.num_actions
    _temp_agent = None 
    q_value_history = np.zeros((num_simulations, num_trials, num_actions))
    
    #SIMULATING CONFUSION/RE-LEARNING
    PRE_TRAIN_TRIALS = 75 #pre-training on the wrong arm with 75 trials
    
    for i in range(num_simulations):
        agent = model_class(**params_dict)
        agent.reset()
        
        #Build up incorrect q-values to simulate reward arm switching - i.e., past trials were all good, but suddenly this same behaviour is bad!
        high_arm_pretrain = 0
        for t_pre in range(PRE_TRAIN_TRIALS):
            #Pass trial number
            action_taken = agent.choose_action(t_pre)
            reward_prob = prob_low 
            if action_taken == high_arm_pretrain:
                reward_prob = prob_high
            reward = 1 if random.uniform(0, 1) < reward_prob else 0
            agent.update_q_value(action_taken, reward, t_pre)
            
        #SIMULATION
        high_arm_sim = 1 #simulate that reward arm is arm 1 
        for t in range(num_trials):
            action_taken = agent.choose_action(t)
            reward_prob = prob_low
            if action_taken == high_arm_sim:
                reward_prob = prob_high
            reward = 1 if random.uniform(0, 1) < reward_prob else 0
            agent.update_q_value(action_taken, reward, t)
            
            if action_taken == high_arm_sim:
                agent_behavior[i, t] = 1
                
            #Store Q-values
            q_value_history[i, t, :] = agent.q_values
    
    #Average behaviour and average q-history to see what estimates hover at!
    avg_behavior = np.mean(agent_behavior, axis=0)
    avg_q_history = np.mean(q_value_history, axis=0)
    
    return avg_behavior, avg_q_history

def find_best_model_by_rmse(model_name, mouse_curve_data, num_trials, prob_high, prob_low, num_search_iterations, sims_per_set):
    """
    Fits a model by finding parameters that minimize RMSE.
    This is a simulation-based fit, not a -LL fit.
    """
    print(f"--- Starting RMSE-based Parameter Search for: {model_name} ---")
    
    if model_name not in MODELS:
        print(f"Error: Model '{model_name}' not defined."); return None, None, None
    
    model_info = MODELS[model_name]
    param_ranges = model_info['param_ranges']
    param_names = list(param_ranges.keys())

    best_rmse = float('inf')
    best_params = {}
    best_agent_curve = None 
    best_q_curve = None #also include estimated q-values

    print(f"Testing {num_search_iterations} random parameter combinations...")
    print(f"(Each test runs {sims_per_set} simulations)")
    
    # LOOPING FOR RANDOMIZED SEARCH
    for i in range(num_search_iterations):
        
        #Random set of parameters to test
        params_dict = {}
        for param_name, param_range in param_ranges.items():
            if 'trial' in param_name: # Handle integer params
                params_dict[param_name] = random.randint(param_range[0], param_range[1])
            else:
                params_dict[param_name] = random.uniform(param_range[0], param_range[1])
                
        average_agent_behavior, avg_q_history = simulate_agent_with_params(
            model_name,
            params_dict, #gets dictionary of parameters (random search)
            num_trials,
            prob_high,
            prob_low,
            sims_per_set  #simulation number
        )
        
        #Compare mouse agent with average real mouse
        score = calculate_rmse(average_agent_behavior, mouse_curve_data)
        
        #Check if this is the best score - save the best score only not all of them
        if score < best_rmse: #if lower (therefore better)
            print(f"  Iter {i+1}/{num_search_iterations}: New Best RMSE: {score:.4f}")
            best_rmse = score
            best_params = params_dict
            best_agent_curve = average_agent_behavior # Save the curve
            best_q_curve = avg_q_history  #also same for q-values 


    print("\n--- RMSE Search Complete ---")
    print(f"Best-fitting parameters (minimized RMSE): {best_params}")
    print(f"Best RMSE score: {best_rmse:.4f}")
    
    return best_params, best_agent_curve, best_rmse, best_q_curve

#PLOTTING
def plot_results(mouse_curve_data, agent_curve_data, condition_str, model_name, best_params, rmse, max_trials=None):
    """Generates and saves the final plot."""
    print("Generating final plot...")
    plt.figure(figsize=(12, 6))
    
    plt.plot(np.arange(1, len(mouse_curve_data) + 1), 
             mouse_curve_data, 
             label=f'Average Mouse ({condition_str})',
             linewidth=2, alpha=0.8)

    plt.plot(np.arange(1, len(agent_curve_data) + 1),
             agent_curve_data, 
             label=f'Simulated Agent (RMSE={rmse:.4f})', 
             linestyle='--', linewidth=2)
    
    #Add max_trials to title and filename 
    title_str = f'Mouse Behavior vs. Best-Fit Agent ({model_name})'
    filename_str = f"plot_{model_name}_{condition_str}_fit_RMSE"
    if max_trials is not None:
        title_str += f" (Trials 1-{max_trials})" #formatting
        filename_str += f"_T{max_trials}"

    plt.title(title_str)
    plt.xlabel('Trials Since Switch (blockTrial)')
    plt.ylabel('% "Good" Option Chosen')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    filename = f"{filename_str}.png"
    save_path = os.path.join("plots", filename) 
    plt.savefig(save_path)
    print(f"Plot saved as {save_path}")
    plt.show()

#Q-VALUE HISTORY PLOTTING
def plot_q_value_history(avg_q_history, prob_high, prob_low, condition_str, model_name, max_trials=None): 
    """
    Generates and saves a plot of the agent's Q-value estimates over time.
    """
    print("Generating Q-value history plot...")
    plt.figure(figsize=(12, 6))

    #In our simulation, high_arm_sim = 1 and high_arm_pretrain (bad arm) = 0
    #So, index 1 is the "Good Arm" and index 0 is the "Bad Arm"
    q_value_good_arm = avg_q_history[:, 1]
    q_value_bad_arm = avg_q_history[:, 0]

    #Agent's internal estimates of value
    plt.plot(np.arange(1, len(q_value_good_arm) + 1), 
             q_value_good_arm, 
             label="Agent's Q-Value for 'Good' Arm", 
             color='blue', linestyle='--')
    
    plt.plot(np.arange(1, len(q_value_bad_arm) + 1), 
             q_value_bad_arm, 
             label="Agent's Q-Value for 'Bad' Arm", 
             color='red', linestyle=':')

    #Actual probabilities as horizontal lines
    plt.axhline(y=prob_high, color='blue', linestyle='-', alpha=0.5, 
                label=f"Actual Probability (Good Arm = {prob_high})")
    plt.axhline(y=prob_low, color='red', linestyle='-', alpha=0.5, 
                label=f"Actual Probability (Bad Arm = {prob_low})")

    #Add max_trials to title and filename (formatting) 
    title_str = f"Agent's Q-Value Learning Curve ({model_name}, {condition_str})"
    filename_str = f"plot_{model_name}_{condition_str}_Q_Values"
    if max_trials is not None:
        title_str += f" (Trials 1-{max_trials})"
        filename_str += f"_T{max_trials}"
    

    plt.title(title_str) 
    plt.xlabel('Trials Since Switch (blockTrial)')
    plt.ylabel('Estimated Q-Value (Expected Reward)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    #Save the plot
    filename = f"{filename_str}.png" 
    save_path = os.path.join("plots", filename)
    plt.savefig(save_path)
    print(f"Plot saved as {save_path}")
    plt.show()
    
#CLI FUNCTIONS AND IMPLEMENTATION
def main():
    parser = argparse.ArgumentParser(description="Fit RL models to bandit task data by minimizing RMSE.")
    parser.add_argument('-m', '--model', type=str, required=True, choices=MODELS.keys(), help=f"Model name to run. Choices: {list(MODELS.keys())}")
    parser.add_argument('-c', '--condition', type=str, required=True, choices=['70', '80', '90'], help="High-reward probability (e.g., '70').")
    parser.add_argument('--datafile', type=str, default='bandit_data.csv', help="Path to your data CSV file.")
    parser.add_argument('--search_iterations', type=int, default=50, help="Number of random parameter sets to try.")
    parser.add_argument('--sims_per_set', type=int, default=100, help="Number of simulations to run *for each* parameter set.")
    parser.add_argument('--max_trials', type=int, default=None, help="Truncate analysis to this many trials per block (e.g., 382).")
    args = parser.parse_args()
    
    start_time = time.time()
    
    os.makedirs("plots", exist_ok=True)
    
    #Load and Prep Data
    condition_str, prob_high, prob_low = parse_condition_arg(args.condition)
    data_condition, mouse_curve_data = load_and_prep_data(args.datafile, condition_str, args.max_trials) # <-- MODIFIED
    
    if mouse_curve_data is None:
        print("Data loading failed.")
        return
        
    num_trials = len(mouse_curve_data) #truncated length

    #Find Best Model (by minimizing RMSE)
    best_params, best_agent_curve, best_rmse, best_q_curve = find_best_model_by_rmse(
        args.model,
        mouse_curve_data,
        num_trials,
        prob_high,
        prob_low,
        args.search_iterations,
        args.sims_per_set
    )
    if best_params is None: 
        print("Model fitting failed.")
        return

    # Plot Results
    plot_results(
        mouse_curve_data, 
        best_agent_curve, 
        condition_str, 
        args.model, 
        best_params, 
        best_rmse,
        args.max_trials 
    )
    
    #Plot Q-Value History
    if best_q_curve is not None:
        plot_q_value_history(
            best_q_curve,
            prob_high,
            prob_low,
            condition_str,
            args.model,
            args.max_trials 
        )
    else:
        print("Could not generate Q-value plot (no data).")
    
    print("\n--- Final Results ---")
    print(f"Model: {args.model}, Condition: {condition_str}")
    if args.max_trials is not None:
        print(f"Analysis truncated to {args.max_trials} trials.")
    print(f"Best-Fit Params (from RMSE): {best_params}")
    print(f"Final Behavior Fit (RMSE): {best_rmse:.4f}") 
    print(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()