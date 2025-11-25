document.getElementById('runForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const runBtn = document.getElementById('runBtn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');

    // Reset UI
    runBtn.disabled = true;
    loading.classList.remove('hidden');
    results.classList.remove('hidden');
    document.getElementById('stats').innerHTML = '';
    document.getElementById('behaviorPlot').src = '';
    document.getElementById('qValuePlot').src = '';

    // Gather form data
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());

    try {
        const response = await fetch('/run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'Simulation failed');
        }

        // Update UI with results
        displayStats(result);

        if (result.behavior_plot) {
            document.getElementById('behaviorPlot').src = `/plots/${result.behavior_plot}?t=${Date.now()}`;
        }

        if (result.q_value_plot) {
            document.getElementById('qValuePlot').src = `/plots/${result.q_value_plot}?t=${Date.now()}`;
            document.getElementById('qValuePlot').parentElement.classList.remove('hidden');
        } else {
            document.getElementById('qValuePlot').parentElement.classList.add('hidden');
        }

    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        runBtn.disabled = false;
        loading.classList.add('hidden');
    }
});

function displayStats(result) {
    const statsContainer = document.getElementById('stats');

    const rmseCard = createStatCard('RMSE', result.rmse.toFixed(4));
    statsContainer.appendChild(rmseCard);

    // Format params nicely
    const paramsStr = Object.entries(result.best_params)
        .map(([key, value]) => `${key}: ${typeof value === 'number' ? value.toFixed(4) : value}`)
        .join('<br>');

    const paramsCard = createStatCard('Best Parameters', paramsStr);
    statsContainer.appendChild(paramsCard);
}

function createStatCard(label, value) {
    const div = document.createElement('div');
    div.className = 'stat-card';
    div.innerHTML = `
        <div class="stat-label">${label}</div>
        <div class="stat-value">${value}</div>
    `;
    return div;
}
