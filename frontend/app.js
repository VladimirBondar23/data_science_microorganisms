// ---------- Config ----------
const API_BASE = "http://127.0.0.1:5000"; // change to "/api" if using a Node proxy

// ---------- Elements ----------
const btn = document.getElementById("goBtn");
const resultEl = document.getElementById("result");
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
btn.addEventListener("click", fetchCommunity);

async function fetchCommunity() {
  const level = document.getElementById("levelDropdown").value;
  const env = document.getElementById("environmentInput").value.trim();

  if (!level) return alert("Please select a taxonomy level!");
  if (!env) return alert("Please enter an environment / organism_name!");

  let url = `${API_BASE}/samples/?level=${encodeURIComponent(level)}&organism_name=${encodeURIComponent(env)}`;
  if (latEl.value && lonEl.value) {
    url += `&lat=${encodeURIComponent(latEl.value)}&lon=${encodeURIComponent(lonEl.value)}`;
  }
  if (coordLabelEl.value) {
    // optional convenience param carrying "lat, lon" as a single string
    url += `&coords=${encodeURIComponent(coordLabelEl.value)}`;
  }

  try {
    btn.disabled = true;
    btn.textContent = "Loading...";

    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    // Show raw JSON
    resultEl.textContent = JSON.stringify(data, null, 2);

    // Build chart from numeric entries
    let entries = Object.entries(data).filter(([_, v]) => typeof v === "number");
    entries.sort((a, b) => b[1] - a[1]);
    entries = entries.slice(0, 15); // top 15

    const labels = entries.map(([k]) => k);
    const values = entries.map(([_, v]) => Number(v));
    const chartType = labels.length <= 8 ? "pie" : "bar";

    if (labels.length && values.length) {
      if (chart) chart.destroy();
      chart = new Chart(ctx, {
        type: chartType,
        data: {
          labels,
          datasets: [{ label: `${level} composition`, data: values }]
        },
        options: {
          responsive: true,
          plugins: { legend: { display: true } },
          ...(chartType === "bar" ? { scales: { y: { beginAtZero: true } } } : {})
        }
      });
    }
  } catch (err) {
    console.error(err);
    alert("Failed to fetch data from backend. Make sure FastAPI is running and CORS is enabled.");
  } finally {
    btn.disabled = false;
    btn.textContent = "Get Community Composition";
  }
}
