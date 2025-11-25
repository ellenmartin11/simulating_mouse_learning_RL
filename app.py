from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import sys
import traceback
import numpy as np

#current directory to sys.path to import the script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import fitRLmodel_app_backend as rl_model

app = Flask(__name__)

#plots directory 
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html', models=list(rl_model.MODELS.keys()))

@app.route('/run', methods=['POST'])
def run_simulation():
    try:
        data = request.json
        model_name = data.get('model')
        condition = data.get('condition')
        max_trials = data.get('max_trials')
        search_iterations = int(data.get('search_iterations', 50))
        sims_per_set = int(data.get('sims_per_set', 100))

        if max_trials == '':
            max_trials = None
        else:
            max_trials = int(max_trials)

        #Load and Prep Data
        condition_str, prob_high, prob_low = rl_model.parse_condition_arg(condition)
        # Assuming the data file is in the same directory
        datafile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bandit_data.csv')
        
        data_condition, mouse_curve_data = rl_model.load_and_prep_data(datafile, condition_str, max_trials)
        num_trials = len(mouse_curve_data)

        #Find Best Model
        best_params, best_agent_curve, best_rmse, best_q_curve = rl_model.find_best_model_by_rmse(
            model_name,
            mouse_curve_data,
            num_trials,
            prob_high,
            prob_low,
            search_iterations,
            sims_per_set
        )

        if best_params is None:
            return jsonify({'error': 'Model fitting failed.'}), 500

        #Plot Results
        behavior_plot = rl_model.plot_results(
            mouse_curve_data, 
            best_agent_curve, 
            condition_str, 
            model_name, 
            best_params, 
            best_rmse,
            max_trials,
            show_plot=False
        )

        #Plot Q-Value History
        q_value_plot = None
        if best_q_curve is not None:
            q_value_plot = rl_model.plot_q_value_history(
                best_q_curve,
                prob_high,
                prob_low,
                condition_str,
                model_name,
                max_trials,
                show_plot=False
            )

        return jsonify({
            'success': True,
            'best_params': best_params,
            'rmse': best_rmse,
            'behavior_plot': behavior_plot,
            'q_value_plot': q_value_plot
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/plots/<filename>')
def get_plot(filename):
    return send_from_directory(PLOTS_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
