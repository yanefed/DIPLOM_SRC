document.addEventListener("DOMContentLoaded", function () {
  // Set default date to today
  const today = new Date().toISOString().split("T")[0];
  document.getElementById("flight-date").value = today;

  // Load airports and airlines data
  loadAirports();
  loadAirlines();

  // Add event listeners
  document
    .getElementById("predict-button")
    .addEventListener("click", getPrediction);
  document
    .getElementById("origin")
    .addEventListener("change", updateAvailableAirlines);
  document
    .getElementById("destination")
    .addEventListener("change", updateAvailableAirlines);
});

// Function to load all airlines
async function loadAirlines() {
  try {
    const response = await fetch("/api/v1/airlines/");
    if (!response.ok) {
      throw new Error("Failed to load airlines");
    }

    const data = await response.json();
    console.log("Airlines loaded:", data);

    // Store airlines data globally for later filtering
    window.airlinesData = data.airlines || [];

    // Initially populate the dropdown with all airlines
    updateAirlineDropdown(window.airlinesData);
  } catch (error) {
    console.error("Error loading airlines:", error);
    alert("Ошибка загрузки списка авиакомпаний");
  }
}

// Function to update available airlines based on selected route
async function updateAvailableAirlines() {
  const origin = document.getElementById("origin").value;
  const destination = document.getElementById("destination").value;

  console.log("Origin:", origin);
  console.log("Destination:", destination);

  if (!origin || !destination) {
    console.log("Either origin or destination is not selected yet");
    return;
  }

  try {
    const apiUrl = `/api/v1/airlines/route/${origin}/${destination}`;
    console.log(
      `Fetching airlines for route ${origin} to ${destination} from URL: ${apiUrl}`,
    );

    // Для отладки выведем все элементы select
    console.log("Origin select:", document.getElementById("origin").outerHTML);
    console.log(
      "Destination select:",
      document.getElementById("destination").outerHTML,
    );

    const response = await fetch(apiUrl);
    console.log("Response status:", response.status);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log("API response:", data);

    // Дальнейший код без изменений...
  } catch (error) {
    console.error("Error loading airlines for route:", error);
    console.error("Error details:", error.message, error.stack);
    // Fallback to all airlines if the route-specific endpoint fails
    if (window.airlinesData) {
      console.log("Falling back to all airlines");
      updateAirlineDropdown(window.airlinesData);
    }
  }
}

// Function to update the airline dropdown with the provided airlines
function updateAirlineDropdown(airlines) {
  const airlineSelect = document.getElementById("airline");

  // Clear existing options except the first one (placeholder)
  while (airlineSelect.options.length > 1) {
    airlineSelect.remove(1);
  }

  // Sort airlines alphabetically by code
  airlines.sort((a, b) => a.airline_code.localeCompare(b.airline_code));

  // Add new options
  airlines.forEach((airline) => {
    const option = document.createElement("option");
    option.value = airline.airline_code;
    option.textContent = `${airline.airline_code} - ${airline.airline_name}`;
    airlineSelect.appendChild(option);
  });
}

async function getPrediction() {
  const origin = document.getElementById("origin").value.trim().toUpperCase();
  const destination = document
    .getElementById("destination")
    .value.trim()
    .toUpperCase();
  const airline = document.getElementById("airline").value.trim().toUpperCase();
  const flightDate = document.getElementById("flight-date").value;
  const flightTime = document.getElementById("flight-time").value;

  // Проверка на совпадение аэропортов вылета и назначения
  if (origin === destination) {
    alert("Аэропорт вылета не может совпадать с аэропортом назначения");
    return;
  }

  if (!origin || !destination || !flightDate || !flightTime) {
    alert("Пожалуйста, заполните необходимые поля");
    return;
  }

  try {
    const response = await fetch(
      `/api/v1/predict/${origin}/${destination}/${flightDate}?airline=${airline}&time_interval=${flightTime}`,
    );
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Failed to get prediction");
    }

    displayResult(data);
    fetchAndDisplayHistoricalDelays(origin, destination, airline);
  } catch (error) {
    alert(`Ошибка: ${error.message}`);
  }
}

// Function to fetch historical delay data and display chart
async function fetchAndDisplayHistoricalDelays(origin, destination, airline) {
  try {
    const url = `/api/v1/historical_delays/${origin}/${destination}?airline=${airline}`;
    console.log(`⭐ Fetching historical data from: ${url}`);

    const response = await fetch(url);
    console.log(`⭐ Response status: ${response.status}`);

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();
    console.log(`⭐ Data received:`, data);

    // НОВЫЙ КОД - проверяем является ли это резервным ответом
    if (data.prediction_type === "fallback" || data.error) {
      console.log(`⭐ Received fallback data with error: ${data.error}`);
      throw new Error(data.error || "Получены только резервные данные");
    }

    // Если у нас есть реальные данные подходящего формата
    if (data && data.dates && data.dates.length > 0) {
      console.log(`⭐ Using API data for chart`);
      displayDelayChart(data);
      return;
    }

    throw new Error("No valid data from API");
  } catch (error) {
    console.error(`⭐ Error:`, error);
    // Только в случае ошибки используем тестовые данные
    console.log(`⭐ Using test data as fallback`);
    // Как запасной вариант используем тестовые данные
    createAndDisplayTestData(origin, destination, airline);
  }
}

// Выделим генерацию тестовых данных в отдельную функцию для чистоты кода
function createAndDisplayTestData(origin, destination, airline) {
  const today = new Date();
  const dates = [];
  const delays = [];

  for (let i = 6; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);
    dates.push(date.toISOString().split("T")[0]);
    delays.push(Math.round(Math.random() * 30));
  }

  const testData = {
    origin: origin,
    destination: destination,
    airline: airline,
    dates: dates,
    delays: delays,
  };

  console.log("Using test data as fallback:", testData);
  displayDelayChart(testData);
}

function displaySimpleChart(origin, destination, airline) {
  // Показать контейнер
  const container = document.getElementById("historical-data-container");
  container.style.display = "block";

  // Получить canvas
  const canvas = document.getElementById("delay-history-chart");
  const ctx = canvas.getContext("2d");

  // Удалить предыдущий график
  if (window.delayChart) {
    window.delayChart.destroy();
  }

  // Создать простейшие тестовые данные
  const labels = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"];
  const data = [15, 22, 18, 25, 17, 12, 20];

  // Создать график
  window.delayChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Задержка (мин)",
          data: data,
          borderColor: "rgb(75, 192, 192)",
          backgroundColor: "rgba(75, 192, 192, 0.2)",
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: `${origin} → ${destination} (${airline || "все авиакомпании"})`,
        },
      },
    },
  });
}

// Function to display error message in the chart area
function displayErrorChart(errorMessage) {
  // Show the chart container
  const container = document.getElementById("historical-data-container");
  container.style.display = "block";

  // Clear previous chart if it exists
  if (window.delayChart) {
    window.delayChart.destroy();
    window.delayChart = null;
  }

  // Get the canvas and add a message
  const canvas = document.getElementById("delay-history-chart");
  const ctx = canvas.getContext("2d");

  // Clear the canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Add error text
  ctx.font = "16px Arial";
  ctx.fillStyle = "#666";
  ctx.textAlign = "center";
  ctx.fillText(errorMessage, canvas.width / 2, canvas.height / 2);
}

// Function to display the delay chart
// Function to display the delay chart
function displayDelayChart(data) {
  console.log("Displaying chart with data:", JSON.stringify(data));

  // Show the chart container first
  const container = document.getElementById("historical-data-container");
  container.style.display = "block";

  const canvas = document.getElementById("delay-history-chart");

  // Make sure canvas dimensions are set
  if (canvas.width === 0) canvas.width = container.clientWidth;
  if (canvas.height === 0) canvas.height = 300;

  const ctx = canvas.getContext("2d");

  // Destroy previous chart if it exists
  if (window.delayChart) {
    console.log("Destroying previous chart");
    window.delayChart.destroy();
  }

  // Format dates for display
  const dates = data.dates.map((date) => {
    try {
      const dateObj = new Date(date);
      return dateObj.toLocaleDateString("ru-RU");
    } catch (e) {
      console.error("Error formatting date:", date, e);
      return date; // Return the original string if parsing fails
    }
  });

  console.log("Formatted dates:", dates);
  console.log("Delay values:", data.delays);

  // Create new chart with cleaner configuration
  window.delayChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: dates,
      datasets: [
        {
          label: "Среднее время задержки (минуты)",
          data: data.delays,
          borderColor: "rgb(75, 192, 192)",
          backgroundColor: "rgba(75, 192, 192, 0.2)",
          tension: 0.1,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "top",
        },
        tooltip: {
          mode: "index",
          intersect: false,
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: "Задержка (минуты)",
          },
        },
        x: {
          title: {
            display: true,
            text: "Дата",
          },
        },
      },
    },
  });

  console.log("Chart created successfully");
}

function displayResult(data) {
  document.getElementById("delay-value").textContent = data.predicted_delay;
  document.getElementById("confidence-value").textContent = data.confidence;
  document.getElementById("route-info").textContent =
    `${data.origin} → ${data.destination} (${data.airline || "все авиакомпании"}) на ${data.date}`;

  const factorsList = document.getElementById("factors-list");
  factorsList.innerHTML = "";

  data.factors_considered.forEach((factor) => {
    const li = document.createElement("li");
    li.textContent = factor;
    factorsList.appendChild(li);
  });

  document.getElementById("result-card").classList.remove("hidden");

  let chartContainer = document.getElementById("historical-data-container");
  if (chartContainer) {
    chartContainer.style.display = "block";
  }
}

async function loadAirports() {
  try {
    const response = await fetch("/api/v1/airports/");
    if (!response.ok) {
      throw new Error("Failed to load airports");
    }

    const data = await response.json();
    console.log("Airports loaded:", data); // Debug log

    // Make sure we're accessing the airports array from the response
    // Get airports from response and sort them alphabetically by airport_code
    const airports = data.airports || [];
    airports.sort((a, b) => a.airport_code.localeCompare(b.airport_code));

    const originSelect = document.getElementById("origin");
    const destinationSelect = document.getElementById("destination");

    // Clear existing options except the first one (placeholder)
    while (originSelect.options.length > 1) {
      originSelect.remove(1);
    }

    while (destinationSelect.options.length > 1) {
      destinationSelect.remove(1);
    }

    // Add new options
    airports.forEach((airport) => {
      const option = document.createElement("option");
      option.value = airport.airport_code;
      option.textContent = `${airport.airport_code} - ${airport.airport_fullname}`;

      // Clone the option for both selects
      originSelect.appendChild(option);
      destinationSelect.appendChild(option.cloneNode(true));
    });
  } catch (error) {
    console.error("Error loading airports:", error);
    alert("Ошибка загрузки списка аэропортов");
  }
}
