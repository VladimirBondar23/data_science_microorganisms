// ---------- Config ----------
const API_BASE = "http://127.0.0.1:5002"; // change to "/api" if using a Node proxy

// ---------- Elements ----------
const btn = document.getElementById("goBtn");
const ctx = document.getElementById("chart").getContext("2d");
const latEl = document.getElementById("latInput");
const lonEl = document.getElementById("lonInput");
const coordLabelEl = document.getElementById("coordLabel");
let chart;

// ---------- Map (Leaflet) ----------
const map = L.map("map").setView([31.5, 34.75], 7); // default: Israel
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

let marker;

function updateCoordLabel(lat, lon) {
  const label = `${Number(lat).toFixed(6)}, ${Number(lon).toFixed(6)}`;
  coordLabelEl.value = label; // show "lat, lon"
  if (marker) marker.bindPopup(label).openPopup(); // popup on marker
}

function setCoords(lat, lon) {
  latEl.value = Number(lat).toFixed(6);
  lonEl.value = Number(lon).toFixed(6);

  if (marker) {
    marker.setLatLng([lat, lon]);
  } else {
    marker = L.marker([lat, lon], { draggable: true }).addTo(map);
    marker.on("dragend", () => {
      const { lat, lng } = marker.getLatLng();
      setCoords(lat, lng);
    });
  }
  updateCoordLabel(lat, lon);
}

map.on("click", (e) => setCoords(e.latlng.lat, e.latlng.lng));

document.getElementById("useLocation").addEventListener("click", () => {
  if (!navigator.geolocation) return alert("Geolocation is not supported");
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      const { latitude, longitude } = pos.coords;
      map.setView([latitude, longitude], 10);
      setCoords(latitude, longitude);
    },
    () => alert("Unable to get your location")
  );
});

// ---------- Fetch backend + chart ----------
// Prevent form submission from reloading the page and use fetchCommunity
const form = document.querySelector('.input-form');
form.addEventListener('submit', function(e) {
  e.preventDefault();
  fetchCommunity();
});

async function fetchCommunity() {
  const levelDropdown = document.querySelector('#levelDropdown');
  const environmentInput = document.querySelector('#environmentInput');
  const level = levelDropdown ? levelDropdown.value : '';
  const env = environmentInput ? environmentInput.value.trim() : '';

  if (!level) return alert("Please select a taxonomy level!");
  if (!env) return alert("Please enter an environment / organism_name!");

  let url = `${API_BASE}/samples/?level=${encodeURIComponent(level)}&organism_name=${encodeURIComponent(env)}`;
  if (latEl && lonEl && latEl.value && lonEl.value) {
    url += `&lat=${encodeURIComponent(latEl.value)}&lon=${encodeURIComponent(lonEl.value)}`;
  }
  if (latEl && lonEl && latEl.value && lonEl.value) {
    // Format as 'lat N lon E'
    const lat = Number(latEl.value).toFixed(2);
    const lon = Number(lonEl.value).toFixed(2);
    const coordStr = `${lat} N ${lon} E`;
    url += `&coords=${encodeURIComponent(coordStr)}`;
  }

  alert("Fetching data from backend: " + url);

  try {
    btn.disabled = true;
    btn.textContent = "Loading...";

    const res = await fetch(url);
    if (!res.ok) {
      console.error('Fetch failed:', url, 'Status:', res.status, 'Response:', await res.text());
      throw new Error(`HTTP ${res.status}`);
    }
    const data = await res.json();

    // Extract and transform 'average' object for chart data
    let entries = [];
    if (data && typeof data === 'object' && data.average && typeof data.average === 'object') {
      entries = Object.entries(data.average).filter(([_, v]) => typeof v === 'number' && v !== null && v !== undefined);
    } else {
      // fallback: try to use numeric entries from root object
      entries = Object.entries(data).filter(([_, v]) => typeof v === 'number' && v !== null && v !== undefined);
    }
    entries.sort((a, b) => b[1] - a[1]);
    entries = entries.slice(0, 15); // top 15

    // Filter out any entries with null/undefined values before rendering
    const safeEntries = entries.filter(([_, v]) => v !== null && v !== undefined);
    const labels = safeEntries.map(([k]) => k);
    const values = safeEntries.map(([_, v]) => Number(v));
    renderCharts(safeEntries.map(([label, value]) => ({ label, value })));
  } catch (err) {
    console.error(err);
    alert("Failed to fetch data from backend. Make sure FastAPI is running and CORS is enabled.");
  } finally {
    btn.disabled = false;
    btn.textContent = "Get Community Composition";
  }
}

// Chart.js chart instances
let barChartInstance = null;
let pieChartInstance = null;

function renderCharts(data) {
  const labels = data.map(item => item.label);
  const values = data.map(item => item.value);

  // Bar chart
  const barCtx = document.getElementById('chart').getContext('2d');
  if (barChartInstance) barChartInstance.destroy();
  barChartInstance = new Chart(barCtx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Abundance',
        data: values,
        backgroundColor: '#27ae60',
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        title: { display: false },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { display: false }, // Remove x-axis labels
        },
        y: { beginAtZero: true, grid: { display: true } },
      },
    },
  });

  // Pie chart
  const pieCtx = document.getElementById('pieChart').getContext('2d');
  if (pieChartInstance) pieChartInstance.destroy();
  pieChartInstance = new Chart(pieCtx, {
    type: 'pie',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: [
          '#27ae60', '#2ecc71', '#3498db', '#e67e22', '#e74c3c', '#9b59b6', '#f1c40f', '#34495e', '#95a5a6', '#16a085'
        ],
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'right' }, // Move legend to the right
        title: { display: false },
      },
    },
  });
}

