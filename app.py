"""
AI City Digital Twin – Smart Environmental Analysis System
Flask application entrypoint: model lifecycle, REST APIs, persistence, and PDF export.
"""

import json
import logging
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from flask import Flask, Response, jsonify, render_template, request
from flask_sqlalchemy import SQLAlchemy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

try:
    import ollama
except ImportError:  # pragma: no cover - defensive for constrained environments
    ollama = None

HTML = None
LOGGER = logging.getLogger("ai_city_twin")


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "sample_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
DB_PATH = os.path.join(INSTANCE_DIR, "app.db")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(INSTANCE_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


def load_weasyprint_html():
    """
    Lazily import WeasyPrint so Windows environments without GTK/Pango can still boot the API.
    """
    global HTML
    try:
        if HTML is None:
            from weasyprint import HTML as WeasyHTML  # type: ignore

            HTML = WeasyHTML  # pragma: no cover
        return HTML
    except Exception as exc:  # pragma: no cover - platform-specific native deps
        LOGGER.warning("WeasyPrint unavailable: %s", exc)
        HTML = None
        return None


class Simulation(db.Model):
    """
    Persisted scenario snapshot for the My Simulations panel (max 8 rows enforced in code).
    """

    __tablename__ = "simulations"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    city = db.Column(db.String(64), nullable=False)
    payload_json = db.Column(db.Text, nullable=False)

    def to_dict(self):
        """Serialize ORM row for JSON responses."""
        try:
            payload = json.loads(self.payload_json)
        except json.JSONDecodeError:
            payload = {}
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "city": self.city,
            "payload": payload,
        }


# Global model reference loaded once at import/startup.
RF_MODEL = None
FEATURE_COLUMNS = [
    "temperature",
    "traffic_density",
    "green_cover",
    "humidity",
    "wind_speed",
]


def ensure_database_tables():
    """
    Create SQLite tables if they do not exist yet.
    """
    try:
        with app.app_context():
            db.create_all()
    except Exception as exc:  # pragma: no cover - startup safety
        app.logger.error("Database initialization failed: %s", exc)


def load_or_train_model():
    """
    Load the RandomForestRegressor from disk or train on embedded CSV and persist pickle.
    Returns the fitted model or None on failure.
    """
    global RF_MODEL
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as model_file:
                RF_MODEL = pickle.load(model_file)
            app.logger.info("Loaded RandomForest model from %s", MODEL_PATH)
            return RF_MODEL

        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Missing training data at {DATA_PATH}")

        frame = pd.read_csv(DATA_PATH)
        missing = [c for c in FEATURE_COLUMNS + ["aqi"] if c not in frame.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        x_matrix = frame[FEATURE_COLUMNS].values
        y_vector = frame["aqi"].values

        x_train, x_test, y_train, y_test = train_test_split(
            x_matrix, y_vector, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)

        score = model.score(x_test, y_test)
        app.logger.info("RandomForest trained. Hold-out R^2 ~= %.4f", score)

        with open(MODEL_PATH, "wb") as model_file:
            pickle.dump(model, model_file)

        RF_MODEL = model
        return RF_MODEL
    except Exception as exc:
        app.logger.exception("Model load/train failed: %s", exc)
        RF_MODEL = None
        return None


def predict_aqi(feature_dict):
    """
    Predict AQI from a dictionary of feature values using the global model.
    """
    try:
        if RF_MODEL is None:
            raise RuntimeError("Model not available")
        vector = np.array(
            [
                [
                    float(feature_dict["temperature"]),
                    float(feature_dict["traffic_density"]),
                    float(feature_dict["green_cover"]),
                    float(feature_dict["humidity"]),
                    float(feature_dict["wind_speed"]),
                ]
            ]
        )
        prediction = float(RF_MODEL.predict(vector)[0])
        return max(0.0, min(500.0, prediction))
    except Exception as exc:
        app.logger.error("Prediction error: %s", exc)
        raise


def call_ollama_chat(system_prompt, user_prompt, model_name="llama3.2"):
    """
    Invoke a local Ollama chat completion with graceful degradation.
    Tries llama3.2 first, then phi3 as a fallback when the primary model is unavailable.
    """
    try:
        if ollama is None:
            return (
                "Ollama Python client is not installed. "
                "Install dependencies and run `ollama serve` with model llama3.2."
            )

        for candidate in [model_name, "llama2", "llama3.2:3b"]:
            try:
                response = ollama.chat(
                    model=candidate,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                if isinstance(response, dict):
                    message = response.get("message")
                else:
                    message = getattr(response, "message", None)

                if isinstance(message, dict):
                    content = (message.get("content") or "").strip()
                else:
                    content = (getattr(message, "content", "") or "").strip()
                if content:
                    return content
            except Exception as inner_exc:
                app.logger.warning(
                    "Ollama model %s failed, trying fallback: %s", candidate, inner_exc
                )
                continue

        return (
            "The model returned an empty response. Try again or check `ollama list` "
            "for llama3.2 or phi3."
        )
    except Exception as exc:
        app.logger.warning("Ollama chat failed: %s", exc)
        return (
            "Unable to reach Ollama on port 11434. Start the daemon with "
            "`ollama serve` and pull a model: `ollama pull llama3.2` or `ollama pull phi3`. "
            f"Technical detail: {exc}"
        )


def trim_simulations_to_eight():
    """
    Keep only the 8 most recent simulation rows.
    """
    try:
        rows = (
            Simulation.query.order_by(Simulation.created_at.desc())
            .offset(8)
            .all()
        )
        for row in rows:
            db.session.delete(row)
        db.session.commit()
    except Exception as exc:
        app.logger.error("Failed trimming simulations: %s", exc)
        db.session.rollback()


def build_plotly_payload(current_aqi, improved_aqi):
    """
    Create Plotly figure JSON for bar and gauge comparison charts.
    """
    try:
        bar_fig = go.Figure(
            data=[
                go.Bar(
                    x=["Current AQI", "Improved AQI"],
                    y=[current_aqi, improved_aqi],
                    marker_color=["#f97316", "#22c55e"],
                )
            ]
        )
        bar_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=20, t=40, b=40),
            yaxis_title="AQI",
            height=320,
        )

        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=improved_aqi,
                delta={"reference": current_aqi},
                gauge={
                    "axis": {"range": [0, 300]},
                    "bar": {"color": "#22c55e"},
                    "steps": [
                        {"range": [0, 50], "color": "#14532d"},
                        {"range": [50, 100], "color": "#166534"},
                        {"range": [100, 150], "color": "#854d0e"},
                        {"range": [150, 200], "color": "#9a3412"},
                        {"range": [200, 300], "color": "#7f1d1d"},
                    ],
                },
            )
        )
        gauge_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(l=20, r=20, t=50, b=20),
        )

        return {
            "bar": json.loads(bar_fig.to_json()),
            "gauge": json.loads(gauge_fig.to_json()),
        }
    except Exception as exc:
        app.logger.error("Plotly build failed: %s", exc)
        raise


def compute_impact_metrics(current_aqi, improved_aqi, green_delta, traffic_delta):
    """
    Derive illustrative health and environmental co-benefits from AQI deltas.
    """
    try:
        delta = max(0.0, float(current_aqi) - float(improved_aqi))
        respiratory = int(round(delta * 3.2))
        co2_saved_kg = round(delta * 1.8 + abs(traffic_delta) * 0.02, 2)
        env_score = max(
            0,
            min(
                100,
                int(
                    round(
                        55
                        + min(20, green_delta * 1.2)
                        + min(15, delta * 0.4)
                        - max(0, -traffic_delta) * 0.01
                    )
                ),
            ),
        )
        return {
            "respiratory_cases_prevented": respiratory,
            "co2_saved_kg": co2_saved_kg,
            "environment_score": env_score,
        }
    except Exception as exc:
        app.logger.error("Impact metrics failed: %s", exc)
        return {
            "respiratory_cases_prevented": 0,
            "co2_saved_kg": 0.0,
            "environment_score": 0,
        }


@app.route("/")
def landing_page():
    """
    Render the premium landing experience.
    """
    try:
        return render_template("index.html")
    except Exception as exc:
        app.logger.exception("Landing render failed: %s", exc)
        return "Landing page unavailable.", 500


@app.route("/dashboard")
def dashboard_page():
    """
    Render the interactive dashboard (standalone full page or iframe target).
    """
    try:
        return render_template("dashboard.html")
    except Exception as exc:
        app.logger.exception("Dashboard render failed: %s", exc)
        return "Dashboard unavailable.", 500


@app.route("/api/city-defaults", methods=["GET"])
def city_defaults():
    """
    Return realistic slider defaults for each supported city.
    """
    try:
        data = {
            "Indore": {
                "temperature": 32,
                "traffic_density": 4500,
                "green_cover": 28,
                "humidity": 55,
                "wind_speed": 12,
            },
            "Ujjain": {
                "temperature": 34,
                "traffic_density": 8800,
                "green_cover": 44,
                "humidity": 48,
                "wind_speed": 10,
            },
            "Delhi": {
                "temperature": 38,
                "traffic_density": 12000,
                "green_cover": 15,
                "humidity": 40,
                "wind_speed": 8,
            },
            "Mumbai": {
                "temperature": 30,
                "traffic_density": 9000,
                "green_cover": 22,
                "humidity": 78,
                "wind_speed": 15,
            },
            "Bhopal": {
                "temperature": 33,
                "traffic_density": 3800,
                "green_cover": 35,
                "humidity": 52,
                "wind_speed": 11,
            },
        }
        return jsonify({"ok": True, "cities": data})
    except Exception as exc:
        app.logger.error("city-defaults error: %s", exc)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Predict AQI from posted environmental features.
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        current = {key: body.get(key) for key in FEATURE_COLUMNS}
        for key in FEATURE_COLUMNS:
            if current[key] is None:
                return jsonify({"ok": False, "error": f"Missing {key}"}), 400
        aqi = predict_aqi(current)
        return jsonify({"ok": True, "aqi": round(aqi, 2)})
    except Exception as exc:
        app.logger.error("predict error: %s", exc)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """
    Run an improved scenario: moderate traffic and boost green cover for uplift preview.
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        base = {key: body.get(key) for key in FEATURE_COLUMNS}
        for key in FEATURE_COLUMNS:
            if base[key] is None:
                return jsonify({"ok": False, "error": f"Missing {key}"}), 400

        current_aqi = predict_aqi(base)

        improved = dict(base)
        improved["traffic_density"] = max(500, float(improved["traffic_density"]) * 0.82)
        improved["green_cover"] = min(95, float(improved["green_cover"]) + 6)
        improved["wind_speed"] = min(40, float(improved["wind_speed"]) + 2)

        improved_aqi = predict_aqi(improved)

        charts = build_plotly_payload(current_aqi, improved_aqi)
        traffic_delta = float(improved["traffic_density"]) - float(base["traffic_density"])
        green_delta = float(improved["green_cover"]) - float(base["green_cover"])
        impact = compute_impact_metrics(current_aqi, improved_aqi, green_delta, traffic_delta)

        return jsonify(
            {
                "ok": True,
                "current_aqi": round(current_aqi, 2),
                "improved_aqi": round(improved_aqi, 2),
                "improved_features": improved,
                "charts": charts,
                "impact": impact,
            }
        )
    except Exception as exc:
        app.logger.error("simulate error: %s", exc)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/recommendations", methods=["POST"])
def api_recommendations():
    """
    Ask Ollama for structured mitigation guidance for the active scenario.
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        city = body.get("city", "Ujjain")
        features = body.get("features", {})
        current_aqi = body.get("current_aqi")
        improved_aqi = body.get("improved_aqi")

        system_prompt = (
            "You are an environmental advisor for Indian smart cities. "
            "Respond with concise bullet points (max 6) focusing on air quality, "
            "traffic, green cover, and public health. Mention Simhastha Kumbh readiness when city is Ujjain."
        )
        user_prompt = (
            f"City: {city}. Environmental inputs: {json.dumps(features)}. "
            f"Current modeled AQI: {current_aqi}. After simulated interventions AQI: {improved_aqi}. "
            "List prioritized actions for city administrators."
        )

        text = call_ollama_chat(system_prompt, user_prompt)
        return jsonify({"ok": True, "text": text})
    except Exception as exc:
        app.logger.error("recommendations error: %s", exc)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Sidebar assistant: free-form questions grounded in the scenario context.
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        question = (body.get("message") or "").strip()
        context = body.get("context", {})
        if not question:
            return jsonify({"ok": False, "error": "Empty message"}), 400

        system_prompt = (
            "You are an AI assistant embedded in the AI City Digital Twin dashboard. "
            "Answer using plain language, cite Indian urban context when relevant, "
            "and stay practical for students and city planners."
        )
        user_prompt = f"Scenario context: {json.dumps(context)}. User question: {question}"
        text = call_ollama_chat(system_prompt, user_prompt)
        return jsonify({"ok": True, "reply": text})
    except Exception as exc:
        app.logger.error("chat error: %s", exc)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/simulations", methods=["GET", "POST"])
def api_simulations():
    """
    GET: list saved simulations. POST: persist a new snapshot (max 8 retained).
    """
    try:
        if request.method == "GET":
            rows = Simulation.query.order_by(Simulation.created_at.desc()).limit(8).all()
            return jsonify({"ok": True, "items": [row.to_dict() for row in rows]})

        body = request.get_json(force=True, silent=True) or {}
        city = body.get("city", "Ujjain")
        payload = body.get("payload", {})
        record = Simulation(city=city, payload_json=json.dumps(payload))
        db.session.add(record)
        db.session.commit()
        trim_simulations_to_eight()
        return jsonify({"ok": True, "saved": record.to_dict()})
    except Exception as exc:
        db.session.rollback()
        app.logger.error("simulations error: %s", exc)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/simulations/<int:sim_id>", methods=["GET"])
def api_simulation_detail(sim_id):
    """
    Fetch a single saved simulation by identifier.
    """
    try:
        row = db.session.get(Simulation, sim_id)
        if not row:
            return jsonify({"ok": False, "error": "Not found"}), 404
        return jsonify({"ok": True, "item": row.to_dict()})
    except Exception as exc:
        app.logger.error("simulation detail error: %s", exc)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/report.pdf", methods=["POST"])
def api_report_pdf():
    """
    Generate a PDF report via WeasyPrint including inputs, charts snapshot, and AI text.
    """
    try:
        weasy_html = load_weasyprint_html()
        if weasy_html is None:
            return jsonify(
                {
                    "ok": False,
                    "error": (
                        "WeasyPrint native libraries missing. On Windows install GTK3 runtime "
                        "per WeasyPrint docs, or generate PDF on WSL/Linux."
                    ),
                }
            ), 500

        body = request.get_json(force=True, silent=True) or {}
        html_content = body.get("html")
        if not html_content:
            return jsonify({"ok": False, "error": "Missing html payload"}), 400

        pdf_bytes = weasy_html(string=html_content, base_url=request.host_url).write_pdf()
        return Response(
            pdf_bytes,
            mimetype="application/pdf",
            headers={"Content-Disposition": "attachment; filename=city-twin-report.pdf"},
        )
    except Exception as exc:
        app.logger.error("PDF export error: %s", exc)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.errorhandler(404)
def handle_not_found(err):
    """
    Friendly JSON/HTML 404 handler.
    """
    try:
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "error": "Not found"}), 404
        return "Page not found. Return to <a href='/'>home</a>.", 404
    except Exception:
        return "Not found", 404


@app.errorhandler(500)
def handle_server_error(err):
    """
    Friendly JSON 500 handler for API routes.
    """
    try:
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "error": "Internal server error"}), 500
        return "Server error", 500
    except Exception:
        return "Server error", 500


# Initialize database, tables, and model before first request.
ensure_database_tables()
load_or_train_model()


if __name__ == "__main__":
    try:
        app.run(debug=True, host="0.0.0.0", port=5000)
    except Exception as exc:
        print(f"Failed to start Flask server: {exc}")
