document.addEventListener('DOMContentLoaded', function() {
    // Set default date to today
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('flight-date').value = today;

    // Add event listener to the predict button
    document.getElementById('predict-button').addEventListener('click', getPrediction);
});

async function getPrediction() {
    const origin = document.getElementById('origin').value.trim().toUpperCase();
    const destination = document.getElementById('destination').value.trim().toUpperCase();
    const flightDate = document.getElementById('flight-date').value;

    if (!origin || !destination || !flightDate) {
        alert('Please fill in all fields');
        return;
    }

    try {
        const response = await fetch(`/api/v1/predict/${origin}/${destination}/${flightDate}`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to get prediction');
        }

        displayResult(data);
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

function displayResult(data) {
    document.getElementById('delay-value').textContent = data.predicted_delay;
    document.getElementById('confidence-value').textContent = data.confidence;
    document.getElementById('route-info').textContent =
        `${data.origin} to ${data.destination} on ${data.date}`;

    const factorsList = document.getElementById('factors-list');
    factorsList.innerHTML = '';

    data.factors_considered.forEach(factor => {
        const li = document.createElement('li');
        li.textContent = factor;
        factorsList.appendChild(li);
    });

    document.getElementById('result-card').classList.remove('hidden');
}