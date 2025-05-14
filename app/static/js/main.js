document.addEventListener('DOMContentLoaded', function() {
    // Set default date to today
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('flight-date').value = today;

    // Load airports and airlines data
    loadAirports();
    loadAirlines();

    // Add event listeners
    document.getElementById('predict-button').addEventListener('click', getPrediction);
    document.getElementById('origin').addEventListener('change', updateAvailableAirlines);
    document.getElementById('destination').addEventListener('change', updateAvailableAirlines);
});

// Function to load all airlines
async function loadAirlines() {
    try {
        const response = await fetch('/api/v1/airlines/');
        if (!response.ok) {
            throw new Error('Failed to load airlines');
        }

        const data = await response.json();
        console.log("Airlines loaded:", data);

        // Store airlines data globally for later filtering
        window.airlinesData = data.airlines || [];
        
        // Initially populate the dropdown with all airlines
        updateAirlineDropdown(window.airlinesData);
    } catch (error) {
        console.error('Error loading airlines:', error);
        alert('Ошибка загрузки списка авиакомпаний');
    }
}

// Function to update available airlines based on selected route
async function updateAvailableAirlines() {
    const origin = document.getElementById('origin').value;
    const destination = document.getElementById('destination').value;
    
    if (!origin || !destination) {
        return;
    }
    
    try {
        // Use the new route-specific endpoint to get airlines for this route
        const response = await fetch(`/api/v1/airlines/route/${origin}/${destination}`);
        if (!response.ok) {
            throw new Error('Failed to load airlines for route');
        }
        
        const data = await response.json();
        console.log(`Airlines for route ${origin} to ${destination}:`, data);
        
        if (data.status === "ok" && data.airlines && data.airlines.length > 0) {
            updateAirlineDropdown(data.airlines);
        } else {
            // Show a message when no airlines operate on this route
            const airlineSelect = document.getElementById('airline');
            while (airlineSelect.options.length > 1) {
                airlineSelect.remove(1);
            }
            const option = document.createElement('option');
            option.value = "";
            option.textContent = "Нет доступных авиакомпаний для этого маршрута";
            option.disabled = true;
            airlineSelect.appendChild(option);
        }
    } catch (error) {
        console.error('Error loading airlines for route:', error);
        // Fallback to all airlines if the route-specific endpoint fails
        if (window.airlinesData) {
            updateAirlineDropdown(window.airlinesData);
        }
    }
}

// Function to update the airline dropdown with the provided airlines
function updateAirlineDropdown(airlines) {
    const airlineSelect = document.getElementById('airline');
    
    // Clear existing options except the first one (placeholder)
    while (airlineSelect.options.length > 1) {
        airlineSelect.remove(1);
    }
    
    // Sort airlines alphabetically by code
    airlines.sort((a, b) => a.airline_code.localeCompare(b.airline_code));
    
    // Add new options
    airlines.forEach(airline => {
        const option = document.createElement('option');
        option.value = airline.airline_code;
        option.textContent = `${airline.airline_code} - ${airline.airline_name}`;
        airlineSelect.appendChild(option);
    });
}

async function getPrediction() {
    const origin = document.getElementById('origin').value.trim().toUpperCase();
    const destination = document.getElementById('destination').value.trim().toUpperCase();
    const airline = document.getElementById('airline').value.trim().toUpperCase();
    const flightDate = document.getElementById('flight-date').value;

    if (!origin || !destination || !flightDate) {
        alert('Пожалуйста, заполните необходимые поля');
        return;
    }

    try {
        const response = await fetch(`/api/v1/predict/${origin}/${destination}/${flightDate}?airline=${airline}`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to get prediction');
        }

        displayResult(data);
    } catch (error) {
        alert(`Ошибка: ${error.message}`);
    }
}

function displayResult(data) {
    document.getElementById('delay-value').textContent = data.predicted_delay;
    document.getElementById('confidence-value').textContent = data.confidence;
    document.getElementById('route-info').textContent = 
        `${data.origin} → ${data.destination} (${data.airline || 'все авиакомпании'}) на ${data.date}`;

    const factorsList = document.getElementById('factors-list');
    factorsList.innerHTML = '';

    data.factors_considered.forEach(factor => {
        const li = document.createElement('li');
        li.textContent = factor;
        factorsList.appendChild(li);
    });

    document.getElementById('result-card').classList.remove('hidden');
}

async function loadAirports() {
    try {
        const response = await fetch('/api/v1/airports/');
        if (!response.ok) {
            throw new Error('Failed to load airports');
        }

        const data = await response.json();
        console.log("Airports loaded:", data); // Debug log

        // Make sure we're accessing the airports array from the response
        // Get airports from response and sort them alphabetically by airport_code
        const airports = data.airports || [];
        airports.sort((a, b) => a.airport_code.localeCompare(b.airport_code));

        const originSelect = document.getElementById('origin');
        const destinationSelect = document.getElementById('destination');

        // Clear existing options except the first one (placeholder)
        while (originSelect.options.length > 1) {
            originSelect.remove(1);
        }

        while (destinationSelect.options.length > 1) {
            destinationSelect.remove(1);
        }

        // Add new options
        airports.forEach(airport => {
            const option = document.createElement('option');
            option.value = airport.airport_code;
            option.textContent = `${airport.airport_code} - ${airport.airport_fullname}`;

            // Clone the option for both selects
            originSelect.appendChild(option);
            destinationSelect.appendChild(option.cloneNode(true));
        });
    } catch (error) {
        console.error('Error loading airports:', error);
        alert('Ошибка загрузки списка аэропортов');
    }
}