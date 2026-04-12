/**
 * Dashboard client logic for AI City Digital Twin.
 * Handles sliders, map, Plotly charts, Ollama-backed chat, SQLite history, and PDF export.
 */

(function () {
  "use strict";

  /** @type {Record<string, any> | null} */
  let cityDefaults = null;

  /** @type {number | null} */
  let predictTimer = null;

  /** @type {{ current: number, improved: number, features: any, improvedFeatures: any, impact: any, charts: any } | null} */
  let lastScenario = null;

  /** @type {L.Map | null} */
  let mapInstance = null;

  /**
   * Reads API base prefix for nested deployments.
   * @returns {string}
   */
  function apiBase() {
    try {
      return typeof window.ACD_API_BASE === "string" ? window.ACD_API_BASE : "";
    } catch (err) {
      console.error("apiBase failed", err);
      return "";
    }
  }

  /**
   * Returns current feature vector from sliders.
   * @returns {{ temperature:number, traffic_density:number, green_cover:number, humidity:number, wind_speed:number }}
   */
  function readFeaturesFromDom() {
    try {
      return {
        temperature: Number(document.getElementById("slider-temp").value),
        traffic_density: Number(document.getElementById("slider-traffic").value),
        green_cover: Number(document.getElementById("slider-green").value),
        humidity: Number(document.getElementById("slider-humidity").value),
        wind_speed: Number(document.getElementById("slider-wind").value),
      };
    } catch (err) {
      console.error("readFeaturesFromDom failed", err);
      return {
        temperature: 30,
        traffic_density: 5000,
        green_cover: 30,
        humidity: 50,
        wind_speed: 10,
      };
    }
  }

  /**
   * Updates on-screen value badges next to sliders.
   * @param {{ temperature:number, traffic_density:number, green_cover:number, humidity:number, wind_speed:number }} features
   */
  function updateSliderLabels(features) {
    try {
      document.getElementById("val-temp").textContent = String(features.temperature);
      document.getElementById("val-traffic").textContent = String(features.traffic_density);
      document.getElementById("val-green").textContent = String(features.green_cover);
      document.getElementById("val-humidity").textContent = String(features.humidity);
      document.getElementById("val-wind").textContent = String(features.wind_speed);
    } catch (err) {
      console.error("updateSliderLabels failed", err);
    }
  }

  /**
   * Applies a city’s default parameters to sliders and UI chrome.
   * @param {string} city
   */
  function applyCityDefaults(city) {
    try {
      if (!cityDefaults || !cityDefaults[city]) {
        return;
      }
      const d = cityDefaults[city];
      document.getElementById("slider-temp").value = d.temperature;
      document.getElementById("slider-traffic").value = d.traffic_density;
      document.getElementById("slider-green").value = d.green_cover;
      document.getElementById("slider-humidity").value = d.humidity;
      document.getElementById("slider-wind").value = d.wind_speed;
      updateSliderLabels(readFeaturesFromDom());

      const badge = document.getElementById("simhastha-badge");
      const note = document.getElementById("map-note");
      if (badge) {
        badge.classList.toggle("d-none", city !== "Ujjain");
      }
      if (note) {
        note.textContent =
          city === "Ujjain"
            ? "Ujjain marker: elevated rehearsal traffic, targeted 2028 canopy uplift along Shipra ghats."
            : "";
      }
    } catch (err) {
      console.error("applyCityDefaults failed", err);
    }
  }

  /**
   * Fetches city defaults from Flask and stores them in memory.
   * @returns {Promise<void>}
   */
  async function loadCityDefaults() {
    try {
      const res = await fetch(apiBase() + "/api/city-defaults");
      const data = await res.json();
      if (data && data.ok) {
        cityDefaults = data.cities;
      }
    } catch (err) {
      console.error("loadCityDefaults failed", err);
    }
  }

  /**
   * Calls the prediction endpoint and refreshes the AQI pill.
   * @returns {Promise<number|null>}
   */
  async function refreshPrediction() {
    try {
      const body = readFeaturesFromDom();
      const res = await fetch(apiBase() + "/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!data.ok) {
        throw new Error(data.error || "Predict failed");
      }
      const aqi = data.aqi;
      const pill = document.getElementById("aqi-pill");
      if (pill) {
        pill.textContent = "Current AQI: " + aqi;
      }
      return aqi;
    } catch (err) {
      console.error("refreshPrediction failed", err);
      const pill = document.getElementById("aqi-pill");
      if (pill) {
        pill.textContent = "Current AQI: error";
      }
      return null;
    }
  }

  /**
   * Renders Plotly figures returned from the backend.
   * @param {{ bar:any, gauge:any }} charts
   */
  function renderCharts(charts) {
    try {
      if (!charts || !window.Plotly) {
        return;
      }
      const barEl = document.getElementById("plot-bar");
      const gaugeEl = document.getElementById("plot-gauge");
      if (charts.bar && barEl) {
        Plotly.react(barEl, charts.bar.data, charts.bar.layout, { responsive: true, displaylogo: false });
      }
      if (charts.gauge && gaugeEl) {
        Plotly.react(gaugeEl, charts.gauge.data, charts.gauge.layout, {
          responsive: true,
          displaylogo: false,
        });
      }
    } catch (err) {
      console.error("renderCharts failed", err);
    }
  }

  /**
   * Updates impact tiles with latest scenario metrics.
   * @param {{ respiratory_cases_prevented:number, co2_saved_kg:number, environment_score:number }} impact
   */
  function renderImpact(impact) {
    try {
      if (!impact) {
        return;
      }
      document.getElementById("impact-resp").textContent = String(impact.respiratory_cases_prevented ?? "—");
      document.getElementById("impact-co2").textContent = String(impact.co2_saved_kg ?? "—");
      document.getElementById("impact-score").textContent = String(impact.environment_score ?? "—");
    } catch (err) {
      console.error("renderImpact failed", err);
    }
  }

  /**
   * Requests mitigation copy from the recommendations API.
   * @param {string} city
   * @param {any} features
   * @param {number} currentAqi
   * @param {number} improvedAqi
   * @returns {Promise<void>}
   */
  async function refreshRecommendations(city, features, currentAqi, improvedAqi) {
    const target = document.getElementById("ai-reco-text");
    try {
      if (target) {
        target.textContent = "Contacting local Ollama model…";
      }
      const res = await fetch(apiBase() + "/api/recommendations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          city,
          features,
          current_aqi: currentAqi,
          improved_aqi: improvedAqi,
        }),
      });
      const data = await res.json();
      if (!data.ok) {
        throw new Error(data.error || "Recommendations failed");
      }
      if (target) {
        target.textContent = data.text || "(empty response)";
      }
    } catch (err) {
      console.error("refreshRecommendations failed", err);
      if (target) {
        target.textContent = "Recommendations unavailable. Check Ollama service.";
      }
    }
  }

  /**
   * Runs the simulate pipeline: charts, impact, recommendations.
   * @returns {Promise<void>}
   */
  async function runSimulation() {
    try {
      const city = document.getElementById("city-select").value;
      const body = readFeaturesFromDom();
      const res = await fetch(apiBase() + "/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!data.ok) {
        throw new Error(data.error || "Simulation failed");
      }
      lastScenario = {
        current: data.current_aqi,
        improved: data.improved_aqi,
        features: body,
        improvedFeatures: data.improved_features,
        impact: data.impact,
        charts: data.charts,
      };
      renderCharts(data.charts);
      renderImpact(data.impact);
      await refreshRecommendations(city, body, data.current_aqi, data.improved_aqi);
    } catch (err) {
      console.error("runSimulation failed", err);
      alert("Simulation failed. See console for details.");
    }
  }

  /**
   * Appends a chat bubble to the assistant stream.
   * @param {"user"|"assistant"} role
   * @param {string} text
   */
  function appendChatBubble(role, text) {
    try {
      const stream = document.getElementById("chat-stream");
      if (!stream) {
        return;
      }
      const wrapper = document.createElement("div");
      wrapper.className =
        "mb-2 small rounded-3 px-2 py-1 " +
        (role === "user" ? "bg-emerald-900/60 text-emerald-50 ms-4" : "bg-slate-800 text-slate-100 me-4");
      wrapper.textContent = (role === "user" ? "You: " : "Twin: ") + text;
      stream.appendChild(wrapper);
      stream.scrollTop = stream.scrollHeight;
    } catch (err) {
      console.error("appendChatBubble failed", err);
    }
  }

  /**
   * Sends a chat message to Ollama via Flask.
   * @returns {Promise<void>}
   */
  async function sendChatMessage() {
    try {
      const input = document.getElementById("chat-input");
      const text = (input.value || "").trim();
      if (!text) {
        return;
      }
      input.value = "";
      appendChatBubble("user", text);

      const city = document.getElementById("city-select").value;
      const context = {
        city,
        features: readFeaturesFromDom(),
        lastScenario,
      };

      const res = await fetch(apiBase() + "/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, context }),
      });
      const data = await res.json();
      if (!data.ok) {
        throw new Error(data.error || "Chat failed");
      }
      appendChatBubble("assistant", data.reply || "");
    } catch (err) {
      console.error("sendChatMessage failed", err);
      appendChatBubble("assistant", "Assistant unavailable. Verify Ollama is running.");
    }
  }

  /**
   * Loads the simulations list from SQLite-backed API.
   * @returns {Promise<void>}
   */
  async function refreshSimulationsList() {
    try {
      const holder = document.getElementById("simulations-list");
      if (!holder) {
        return;
      }
      holder.innerHTML = "";
      const res = await fetch(apiBase() + "/api/simulations");
      const data = await res.json();
      if (!data.ok) {
        throw new Error(data.error || "List failed");
      }
      const items = data.items || [];
      if (!items.length) {
        holder.innerHTML =
          '<div class="list-group-item bg-slate-950 text-slate-400">No simulations saved yet.</div>';
        return;
      }
      items.forEach(function (item) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "list-group-item list-group-item-action bg-slate-950 text-slate-100 border-secondary";
        const ts = item.created_at || "";
        btn.textContent = (item.city || "City") + " · " + ts;
        btn.addEventListener("click", function () {
          showSimulationDetail(item);
        });
        holder.appendChild(btn);
      });
    } catch (err) {
      console.error("refreshSimulationsList failed", err);
    }
  }

  /**
   * Shows JSON detail for a saved simulation in the secondary tab.
   * @param {any} item
   */
  function showSimulationDetail(item) {
    try {
      const pre = document.getElementById("simulation-detail");
      const tabTrigger = document.getElementById("tab-detail-tab");
      if (pre) {
        pre.textContent = JSON.stringify(item, null, 2);
      }
      if (tabTrigger && window.bootstrap) {
        const tab = new bootstrap.Tab(tabTrigger);
        tab.show();
      }
    } catch (err) {
      console.error("showSimulationDetail failed", err);
    }
  }

  /**
   * Persists the latest scenario snapshot to the server.
   * @returns {Promise<void>}
   */
  async function saveCurrentSimulation() {
    try {
      if (!lastScenario) {
        alert("Run “Simulate Scenario” before saving.");
        return;
      }
      const city = document.getElementById("city-select").value;
      const payload = {
        ...lastScenario,
        recommendations_text: document.getElementById("ai-reco-text").textContent,
      };
      const res = await fetch(apiBase() + "/api/simulations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ city, payload }),
      });
      const data = await res.json();
      if (!data.ok) {
        throw new Error(data.error || "Save failed");
      }
      await refreshSimulationsList();
      alert("Scenario saved to My Simulations.");
    } catch (err) {
      console.error("saveCurrentSimulation failed", err);
      alert("Unable to save simulation.");
    }
  }

  /**
   * Builds an HTML document string for WeasyPrint export.
   * @returns {string}
   */
  function buildReportHtml() {
    try {
      const city = document.getElementById("city-select").value;
      const feats = readFeaturesFromDom();
      const reco = document.getElementById("ai-reco-text").textContent;
      const when = new Date().toISOString();
      const scenario = lastScenario || {};
      return (
        "<html><head><meta charset='utf-8'><style>body{font-family:system-ui;padding:24px;background:#0f172a;color:#e5e7eb;}h1{color:#6ee7b7;}section{margin-bottom:16px;}pre{background:#020617;padding:12px;border-radius:8px;}</style></head><body>" +
        "<h1>AI City Digital Twin – Environmental Report</h1>" +
        "<section><strong>Generated:</strong> " +
        when +
        "</section>" +
        "<section><strong>City:</strong> " +
        city +
        "</section>" +
        "<section><h2>Inputs</h2><pre>" +
        JSON.stringify(feats, null, 2) +
        "</pre></section>" +
        "<section><h2>Scenario summary</h2><pre>" +
        JSON.stringify(scenario, null, 2) +
        "</pre></section>" +
        "<section><h2>AI suggestions</h2><pre>" +
        reco +
        "</pre></section>" +
        "<section><p>Made for Smart Cities Mission &amp; Simhastha 2028.</p></section>" +
        "</body></html>"
      );
    } catch (err) {
      console.error("buildReportHtml failed", err);
      return "<html><body><p>Unable to compose report.</p></body></html>";
    }
  }

  /**
   * Downloads a PDF by posting HTML to the WeasyPrint endpoint.
   * @returns {Promise<void>}
   */
  async function downloadPdfReport() {
    try {
      const html = buildReportHtml();
      const res = await fetch(apiBase() + "/api/report.pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ html }),
      });
      if (!res.ok) {
        const maybeJson = await res.json().catch(function () {
          return null;
        });
        throw new Error((maybeJson && maybeJson.error) || "PDF endpoint error");
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "ai-city-digital-twin-report.pdf";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("downloadPdfReport failed", err);
      alert("PDF export failed. On Windows, ensure WeasyPrint dependencies are installed.");
    }
  }

  /**
   * Initializes Leaflet map with city markers.
   * @returns {void}
   */
  function initLeafletMap() {
    try {
      if (!window.L) {
        console.error("Leaflet not loaded");
        return;
      }
      const el = document.getElementById("map");
      if (!el || mapInstance) {
        return;
      }
      mapInstance = L.map(el, { scrollWheelZoom: false }).setView([22.5, 79.0], 5);
      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 18,
        attribution: "&copy; OpenStreetMap contributors",
      }).addTo(mapInstance);

      const cities = [
        { name: "Indore", lat: 22.7196, lng: 75.8577 },
        { name: "Ujjain", lat: 23.1765, lng: 75.7885, special: true },
        { name: "Delhi", lat: 28.6139, lng: 77.209 },
        { name: "Mumbai", lat: 19.076, lng: 72.8777 },
        { name: "Bhopal", lat: 23.2599, lng: 77.4126 },
      ];

      cities.forEach(function (c) {
        const color = c.special ? "#22c55e" : "#38bdf8";
        const marker = L.circleMarker([c.lat, c.lng], {
          radius: c.special ? 11 : 8,
          color: "#0f172a",
          weight: 2,
          fillColor: color,
          fillOpacity: 0.9,
        }).addTo(mapInstance);
        const popupText = c.special
          ? "<strong>Ujjain</strong><br/>Simhastha 2028 digital twin rehearsal node."
          : "<strong>" + c.name + "</strong><br/>Click to load defaults.";
        marker.bindPopup(popupText);
        marker.on("click", function () {
          try {
            const select = document.getElementById("city-select");
            select.value = c.name;
            select.dispatchEvent(new Event("change"));
          } catch (err) {
            console.error("marker click failed", err);
          }
        });
      });
    } catch (err) {
      console.error("initLeafletMap failed", err);
    }
  }

  /**
   * Debounced prediction hook for slider movement.
   * @returns {void}
   */
  function schedulePrediction() {
    try {
      if (predictTimer) {
        window.clearTimeout(predictTimer);
      }
      predictTimer = window.setTimeout(function () {
        refreshPrediction();
      }, 250);
    } catch (err) {
      console.error("schedulePrediction failed", err);
    }
  }

  /**
   * Resets the dashboard to Ujjain defaults and clears transient UI.
   * @returns {void}
   */
  function resetDemo() {
    try {
      const select = document.getElementById("city-select");
      select.value = "Ujjain";
      applyCityDefaults("Ujjain");
      lastScenario = null;
      document.getElementById("ai-reco-text").textContent =
        "Run a simulation to generate localized guidance for administrators and field teams.";
      document.getElementById("impact-resp").textContent = "—";
      document.getElementById("impact-co2").textContent = "—";
      document.getElementById("impact-score").textContent = "—";
      document.getElementById("chat-stream").innerHTML = "";
      try {
        if (window.Plotly) {
          Plotly.purge("plot-bar");
          Plotly.purge("plot-gauge");
        }
      } catch (purgeErr) {
        console.warn("Plotly purge skipped", purgeErr);
      }
      refreshPrediction();
    } catch (err) {
      console.error("resetDemo failed", err);
    }
  }

  /**
   * Toggles Bootstrap / Tailwind-friendly light surfaces.
   * @returns {void}
   */
  function toggleDashboardTheme() {
    try {
      const root = document.getElementById("dash-html-root");
      const isDark = root.getAttribute("data-bs-theme") !== "light";
      const next = isDark ? "light" : "dark";
      root.setAttribute("data-bs-theme", next);
      root.classList.toggle("light-surface", next === "light");
      localStorage.setItem("acd-theme", next);
    } catch (err) {
      console.error("toggleDashboardTheme failed", err);
    }
  }

  /**
   * Wires DOM events once the dashboard template is ready.
   * @returns {void}
   */
  function bindDashboardEvents() {
    try {
      const sliders = ["slider-temp", "slider-traffic", "slider-green", "slider-humidity", "slider-wind"];
      sliders.forEach(function (id) {
        const el = document.getElementById(id);
        if (!el) {
          return;
        }
        el.addEventListener("input", function () {
          updateSliderLabels(readFeaturesFromDom());
          schedulePrediction();
        });
      });

      document.getElementById("city-select").addEventListener("change", function (evt) {
        const city = evt.target.value;
        applyCityDefaults(city);
        refreshPrediction();
      });

      document.getElementById("simulate-btn").addEventListener("click", function () {
        runSimulation();
      });

      document.getElementById("save-simulation-btn").addEventListener("click", function () {
        saveCurrentSimulation();
      });

      document.getElementById("chat-send-btn").addEventListener("click", function () {
        sendChatMessage();
      });

      document.getElementById("chat-input").addEventListener("keydown", function (evt) {
        if (evt.key === "Enter") {
          evt.preventDefault();
          sendChatMessage();
        }
      });

      document.getElementById("download-pdf-btn").addEventListener("click", function () {
        downloadPdfReport();
      });

      document.getElementById("reset-demo-btn").addEventListener("click", function () {
        resetDemo();
      });

      document.getElementById("dash-theme-toggle").addEventListener("click", function () {
        toggleDashboardTheme();
      });
    } catch (err) {
      console.error("bindDashboardEvents failed", err);
    }
  }

  /**
   * Entrypoint executed on DOMContentLoaded.
   * @returns {Promise<void>}
   */
  async function bootstrapDashboard() {
    try {
      const stored = localStorage.getItem("acd-theme") || "dark";
      const root = document.getElementById("dash-html-root");
      if (stored === "light") {
        root.setAttribute("data-bs-theme", "light");
        root.classList.add("light-surface");
      } else {
        root.setAttribute("data-bs-theme", "dark");
        root.classList.remove("light-surface");
      }

      await loadCityDefaults();
      bindDashboardEvents();
      const initialCity = document.getElementById("city-select").value;
      applyCityDefaults(initialCity);
      await refreshPrediction();
      initLeafletMap();
      await refreshSimulationsList();
    } catch (err) {
      console.error("bootstrapDashboard failed", err);
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    bootstrapDashboard();
  });
})();
