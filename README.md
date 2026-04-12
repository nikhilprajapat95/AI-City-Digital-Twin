# AI City Digital Twin – Smart Environmental Analysis System

Production-style Flask application for a final-year smart-city project. It combines a Random Forest AQI model, Plotly visualisations, Leaflet mapping, SQLite-backed scenario history, optional WeasyPrint PDF export, and local Ollama reasoning.

## Features

- Premium landing page and full interactive dashboard (Tailwind CSS + Bootstrap 5).
- Random Forest regressor trained on `data/sample_data.csv` (auto-creates `models/random_forest_model.pkl`).
- Scenario simulation with bar + gauge charts (Plotly).
- Ollama-powered recommendations and sidebar chat (`llama3.2`, with automatic fallback attempts for `phi3` / `llama3.2:3b`).
- Impact calculator (illustrative health, CO₂, and environment score metrics).
- Leaflet map of India with city markers; Ujjain highlighted for Simhastha 2028.
- SQLite database at `instance/app.db` for the last eight saved simulations.
- PDF export endpoint using WeasyPrint (requires native OS libraries).

## Quick start

```bash
python -m venv .venv
.\.venv\Scripts\activate          # Windows PowerShell
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000/` for the landing experience, then use **Get Started** to open the dashboard inside a smooth full-screen shell (the parent page does not reload).

## Run with Ollama

1. Install Ollama from [https://ollama.com](https://ollama.com) and ensure the service listens on port **11434** (default).
2. Pull a chat model (primary project default):

   ```bash
   ollama pull llama3.2
   ```

   Optionally also pull a smaller fallback:

   ```bash
   ollama pull phi3
   ```

3. Keep `ollama serve` running in the background while you use the dashboard. Recommendations and the chat assistant call the Flask backend, which forwards prompts to your local models.

If Ollama is offline, the UI still runs; guidance panels show a clear fallback message instead of crashing.

## WeasyPrint notes (PDF export)

WeasyPrint needs Pango/Cairo/GTK-style native libraries. On **Linux** this is usually a single `apt install` per the [WeasyPrint installation guide](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#installation).

On **Windows**, install the recommended GTK runtime for WeasyPrint, or run the app inside **WSL** for simpler dependency resolution. The Flask app starts even if WeasyPrint cannot import; the PDF button then receives a structured error JSON explaining the missing libraries.

## Project layout

```
app.py
templates/
  index.html
  dashboard.html
static/
  css/style.css
  js/script.js
data/sample_data.csv
models/random_forest_model.pkl   # auto-generated
instance/app.db                  # auto-generated
requirements.txt
README.md
```

## Academic honesty

Sample numbers and impact metrics are **illustrative** for demonstration. Replace datasets, validation, and claims with your own measured work before any formal submission.

## Credits footer

Update the `Your Name` placeholder in `templates/index.html` and the dashboard footer note before your viva.
