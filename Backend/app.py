
import os
from dotenv import load_dotenv

load_dotenv()

import hmac
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import pagesizes
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from io import BytesIO
from flask import send_file


from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
import re
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"
from datetime import datetime


app = Flask(__name__)


# PostgreSQL Configuration
# ================= DATABASE CONFIG (Render + Local Compatible) =================

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Render provides DATABASE_URL automatically
DATABASE_URL = os.environ.get("DATABASE_URL")

if DATABASE_URL:
    # Render sometimes provides postgres:// instead of postgresql://
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

    app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL

else:
    # Local development fallback using .env
    DB_USER = os.environ.get("DB_USER")
    DB_PASSWORD = os.environ.get("DB_PASSWORD")
    DB_HOST = os.environ.get("DB_HOST")
    DB_PORT = os.environ.get("DB_PORT", "5432")
    DB_NAME = os.environ.get("DB_NAME")

    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
        raise ValueError("Database environment variables are missing.")

    app.config["SQLALCHEMY_DATABASE_URI"] = (
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

# API Key (required)
API_KEY = os.environ.get("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY environment variable is missing.")

# Initialize Database
db = SQLAlchemy(app)


# Database Model

# this ensures: Same input combination + same material → Cannot be inserted twice.
from sqlalchemy import UniqueConstraint

class Recommendation(db.Model):
    __tablename__ = "recommendation"

    id = db.Column(db.Integer, primary_key=True)

    product_category = db.Column(db.String(50))
    fragility = db.Column(db.String(20))
    shipping_type = db.Column(db.String(20))
    sustainability_priority = db.Column(db.String(20))
    material_name = db.Column(db.String(100))

    predicted_cost = db.Column(db.Float)
    predicted_co2 = db.Column(db.Float)
    suitability_score = db.Column(db.Float)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "product_category",
            "fragility",
            "shipping_type",
            "sustainability_priority",
            "material_name",
            name="unique_recommendation"
        ),
    )

with app.app_context():
    db.create_all()


# Load Dataset & Models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Models
cost_model = joblib.load(os.path.join(BASE_DIR, "models", "cost_model.pkl"))
co2_model = joblib.load(os.path.join(BASE_DIR, "models", "co2_model.pkl"))

# Dataset (now inside data/ folder)
materials_df = pd.read_csv(os.path.join(BASE_DIR, "data", "Ecopack_dataset.csv"))


# Baseline Configuration

BASELINE_MODE = "industry"   # options: "industry"(Dataset Average) or "category"

FEATURE_COLS = [
    "strength",
    "weight_capacity",
    "recyclability_percentage",
    "biodegradability_score"
]


# Industry Baselines (Global Average)

INDUSTRY_BASELINE_CO2 = materials_df["co2_score"].mean()

# Since original dataset may not contain cost column,
# we will define baseline cost as average predicted cost from dataset features
industry_features = materials_df[FEATURE_COLS]
INDUSTRY_BASELINE_COST = cost_model.predict(industry_features).mean()

# Thresholds

GLOBAL_MAX_STRENGTH = materials_df["strength"].max()
STRENGTH_Q75 = materials_df["strength"].quantile(0.75)
STRENGTH_Q50 = materials_df["strength"].quantile(0.50)
WEIGHT_MEDIAN = materials_df["weight_capacity"].median()
BIO_Q70 = materials_df["biodegradability_score"].quantile(0.70)
CO2_Q75 = materials_df["co2_score"].quantile(0.75)

# Validation

def validate_input(data):
    required = [
        "product_category",
        "fragility",
        "shipping_type",
        "sustainability_priority"
    ]

    for field in required:
        if field not in data:
            return False, f"Missing field: {field}"

    return True, None

# Filtering

def apply_filters(df, product_category, fragility):

    filtered = df.copy()
    category_applied = True

    if product_category == "electronics":
        filtered = filtered[
            (filtered["strength"] >= STRENGTH_Q50) &
            (filtered["co2_score"] <= CO2_Q75)
        ]

    elif product_category == "food":
        filtered = filtered[
            filtered["biodegradability_score"] >= BIO_Q70
        ]

    elif product_category == "cosmetics":
        filtered = filtered[
            filtered["weight_capacity"] <= WEIGHT_MEDIAN
        ]
    else:
        # Unknown category → adaptive fallback logic
        filtered = filtered[
            filtered["strength"] >= STRENGTH_Q50
        ]


    if fragility == "high":
        filtered = filtered[filtered["strength"] >= STRENGTH_Q75]
    elif fragility == "medium":
        filtered = filtered[filtered["strength"] >= STRENGTH_Q50]

    return filtered, category_applied

# ML Prediction

def run_predictions(df):

    df = df.copy()

    features = df[[
        "strength",
        "weight_capacity",
        "recyclability_percentage",
        "biodegradability_score"
    ]]

    df["predicted_cost"] = cost_model.predict(features)
    df["predicted_co2"] = co2_model.predict(features)

    return df

# Weight Logic

def get_weights(product_category, sustainability_priority, shipping_type):

    eco_weight = 0.4
    cost_weight = 0.3
    strength_weight = 0.3

    if sustainability_priority == "high":
        eco_weight += 0.2
        cost_weight -= 0.1
    elif sustainability_priority == "low":
        cost_weight += 0.1

    if shipping_type == "international":
        eco_weight += 0.1

    if product_category == "electronics":
        strength_weight += 0.2
        eco_weight -= 0.1
    elif product_category == "cosmetics":
        cost_weight += 0.1
    elif product_category == "food":
        eco_weight += 0.2
        strength_weight -= 0.1

    total = eco_weight + cost_weight + strength_weight

    return eco_weight/total, cost_weight/total, strength_weight/total

# Scoring

def calculate_score(df, eco_w, cost_w, strength_w):

    df = df.copy()

    df["eco_score"] = 1 / (df["predicted_co2"] + 1)
    df["cost_efficiency"] = 1 / (df["predicted_cost"] + 1)
    df["strength_norm"] = df["strength"] / GLOBAL_MAX_STRENGTH

    df["suitability_score"] = (
        eco_w * df["eco_score"] +
        cost_w * df["cost_efficiency"] +
        strength_w * df["strength_norm"]
    )

    return df



def generate_recommendations(product_category, fragility, shipping_type, sustainability_priority):

    filtered_df, _ = apply_filters(materials_df, product_category, fragility)

    if filtered_df.empty:
        return []

    predicted_df = run_predictions(filtered_df)

    eco_w, cost_w, strength_w = get_weights(
        product_category,
        sustainability_priority,
        shipping_type
    )

    scored_df = calculate_score(predicted_df, eco_w, cost_w, strength_w)

    ranked_df = scored_df.sort_values("suitability_score", ascending=False)

    top_results = ranked_df[[
        "material_name",
        "predicted_cost",
        "predicted_co2",
        "suitability_score"
    ]].head(3)

    return top_results.to_dict(orient="records")


# Database Save Logic

# saving reccomendation result to database, if same materials present in dataset for same i/p combination, then it ignores duplicate & contiues without crashing
def save_to_database(product_category, fragility, shipping_type, sustainability_priority, results):

    for item in results:
        record = Recommendation(
            product_category=product_category,
            fragility=fragility,
            shipping_type=shipping_type,
            sustainability_priority=sustainability_priority,
            material_name=item["material_name"],
            predicted_cost=item["predicted_cost"],
            predicted_co2=item["predicted_co2"],
            suitability_score=item["suitability_score"]
        )

        try:
            db.session.add(record)
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            # Duplicate detected → ignore silently
            


def get_category_baseline(product_category):

    filtered_df, _ = apply_filters(materials_df, product_category, fragility="medium")

    if filtered_df.empty:
        return INDUSTRY_BASELINE_COST, INDUSTRY_BASELINE_CO2

    features = filtered_df[FEATURE_COLS]

    baseline_cost = cost_model.predict(features).mean()
    baseline_co2 = co2_model.predict(features).mean()

    return baseline_cost, baseline_co2


def compute_dashboard_data():

    records = Recommendation.query.all()

    if not records:
        return None

    df = pd.DataFrame([{
        "product_category": r.product_category,
        "material_name": r.material_name,
        "predicted_cost": r.predicted_cost,
        "predicted_co2": r.predicted_co2,
        "created_at": r.created_at
    } for r in records])

    df["created_at"] = pd.to_datetime(df["created_at"])



    # Baseline Selection

    if BASELINE_MODE == "industry":

        df["baseline_cost"] = INDUSTRY_BASELINE_COST
        df["baseline_co2"] = INDUSTRY_BASELINE_CO2

    elif BASELINE_MODE == "category":

        baseline_cost_list = []
        baseline_co2_list = []

        for category in df["product_category"]:
            cost_b, co2_b = get_category_baseline(category)
            baseline_cost_list.append(cost_b)
            baseline_co2_list.append(co2_b)

        df["baseline_cost"] = baseline_cost_list
        df["baseline_co2"] = baseline_co2_list

    
    # Compute Metrics
    

    df["co2_reduction_pct"] = (
        (df["baseline_co2"] - df["predicted_co2"]) / df["baseline_co2"]
    ) * 100

    df["cost_savings"] = df["baseline_cost"] - df["predicted_cost"]

    avg_co2_reduction = round(df["co2_reduction_pct"].mean(), 2)
    avg_cost_savings = round(df["cost_savings"].mean(), 2)

    
    # Trend Data (Grouped by Date)

    trend_df = df.groupby(df["created_at"].dt.date).agg({
        "co2_reduction_pct": "mean",
        "cost_savings": "mean"
    }).reset_index()

    trend_df.columns = ["date", "co2_reduction_pct", "cost_savings"]

    co2_trend_fig = px.line(
        trend_df,
        x="date",
        y="co2_reduction_pct",
        title="CO₂ Reduction Trend",
        markers=True
    )

    cost_trend_fig = px.line(
        trend_df,
        x="date",
        y="cost_savings",
        title="Cost Savings Trend",
        markers=True
    )

    
    # Material Usage
    material_usage = (
        df["material_name"]
        .value_counts()
        .reset_index()
    )

    material_usage.columns = ["material_name", "count"]

    bar_fig = px.bar(
        material_usage,
        x="material_name",
        y="count",
        title="Material Usage Trends"
    )

    pie_fig = px.pie(
        material_usage,
        names="material_name",
        values="count",
        title="Material Usage Distribution"
    )

    
    # Ranking Chart (Horizontal Bar)

    ranking_df = df.groupby("material_name").agg({
        "predicted_cost": "mean",
        "predicted_co2": "mean"
    }).reset_index()

    ranking_df["ranking_score"] = (
        (ranking_df["predicted_cost"].max() - ranking_df["predicted_cost"]) +
        (ranking_df["predicted_co2"].max() - ranking_df["predicted_co2"])
    )

    ranking_df = ranking_df.sort_values("ranking_score", ascending=False).head(5)

    ranking_fig = px.bar(
        ranking_df,
        x="ranking_score",
        y="material_name",
        orientation="h",
        title="Top Material Rankings"
    )

    
    # Convert Charts to HTML

    return {
        "avg_co2_reduction": avg_co2_reduction,
        "avg_cost_savings": avg_cost_savings,
        "bar_chart": bar_fig.to_html(full_html=False),
        "pie_chart": pie_fig.to_html(full_html=False),
        "co2_trend_chart": co2_trend_fig.to_html(full_html=False),
        "cost_trend_chart": cost_trend_fig.to_html(full_html=False),
        "ranking_chart": ranking_fig.to_html(full_html=False)
    }



# ============== Frontend Routes ==============

@app.route("/")
def home():
    metrics = compute_dashboard_data()
    return render_template(
        "home.html",
        active_tab="recommend",
        results=None,
        metrics=metrics
    )

@app.route("/recommend", methods=["POST"])
def recommend():

    def validate_custom_category(category):
        if len(category) < 3:
            return False, "Category must be at least 3 characters."

        if not re.match(r"^[A-Za-z ]+$", category):
            return False, "Category can contain only letters and spaces."

        return True, None

    
    # Get Form Data
    
    product_category = request.form.get("product_category")
    other_category = request.form.get("other_category")
    fragility = request.form.get("fragility")
    shipping_type = request.form.get("shipping_type")
    sustainability_priority = request.form.get("sustainability_priority")

    
    # Handle Custom Category
    if product_category == "other":

        if not other_category:
            metrics = compute_dashboard_data()
            return render_template(
                "home.html",
                active_tab="recommend",
                error="Please enter custom category.",
                results=None,
                metrics=metrics
            )

        other_category = other_category.strip().title()

        valid, error = validate_custom_category(other_category)

        if not valid:
            metrics = compute_dashboard_data()
            return render_template(
                "home.html",
                active_tab="recommend",
                error=error,
                results=None,
                metrics=metrics
            )

        # Override
        product_category = other_category

    
    # Basic Validation
    
    if not all([product_category, fragility, shipping_type, sustainability_priority]):
        metrics = compute_dashboard_data()
        return render_template(
            "home.html",
            active_tab="recommend",
            error="All fields are required.",
            results=None,
            metrics=metrics
        )

    
    
    # Generate Recommendations
    results = generate_recommendations(
        product_category,
        fragility,
        shipping_type,
        sustainability_priority
    )

    if results:
        save_to_database(
            product_category,
            fragility,
            shipping_type,
            sustainability_priority,
            results
        )

    metrics = compute_dashboard_data()

    return render_template(
        "home.html",
        active_tab="results",
        results=results,
        metrics=metrics
    )


# REST API Endpoint

@app.route("/api/recommend", methods=["POST"])
def api_recommend():

    key = request.headers.get("x-api-key")
    
    #import hmac
    if not hmac.compare_digest(key or "", API_KEY or ""):
        return jsonify({"status": "error", "message": "Unauthorized"}), 401


    data = request.get_json()

    valid, error = validate_input(data)
    if not valid:
        return jsonify({"status": "error", "message": error}), 400

    product_category = data["product_category"]

    if product_category.lower() == "other" and "other_category" in data:
        product_category = data["other_category"].strip().title()

    results = generate_recommendations(
        product_category,
        data["fragility"],
        data["shipping_type"],
        data["sustainability_priority"]
    )

    if not results:
        return jsonify({"status": "success", "message": "No exact match found", "data": []})

        # Save for API also
    save_to_database(
        product_category,  # use overridden value
        data["fragility"],
        data["shipping_type"],
        data["sustainability_priority"],
        results
    )


    return jsonify({
        "status": "success",
        "message": "Recommendations generated",
        "data": results
    })


# Route for Dashboards
@app.route("/dashboard")
def dashboard():

    metrics = compute_dashboard_data()

    if not metrics:
        return "No recommendation data available yet."

    return render_template(
        "dashboard.html",
        avg_co2_reduction=metrics["avg_co2_reduction"],
        avg_cost_savings=metrics["avg_cost_savings"],
        bar_chart=metrics["bar_chart"],
        pie_chart=metrics["pie_chart"],
        co2_trend_chart=metrics["co2_trend_chart"],
        cost_trend_chart=metrics["cost_trend_chart"],
        ranking_chart=metrics["ranking_chart"]
    )


# ONLY EXCEL EXPORT
@app.route("/export/excel")
def export_excel():

    records = Recommendation.query.all()

    if not records:
        return "No data available"

    df = pd.DataFrame([{
        "Product Category": r.product_category,
        "Material Name": r.material_name,
        "Predicted Cost": r.predicted_cost,
        "Predicted CO2": r.predicted_co2,
        "Suitability Score": r.suitability_score,
        "Created At": r.created_at
    } for r in records])

    output = BytesIO()
    df.to_excel(output, index=False, engine="openpyxl")
    output.seek(0)

    return send_file(
        output,
        download_name="EcoPackAI_Full_Dataset.xlsx",
        as_attachment=True
    )


# ONLY PDF EXPORT
@app.route("/export/pdf")
def export_pdf():

    metrics = compute_dashboard_data()

    if not metrics:
        return "No data available"

    records = Recommendation.query.all()

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=pagesizes.A4)
    elements = []

    styles = getSampleStyleSheet()

    # Title
    elements.append(Paragraph("EcoPackAI – Sustainability Summary Report", styles["Heading1"]))
    elements.append(Spacer(1, 20))

    # Metrics
    elements.append(Paragraph(f"Average CO₂ Reduction: {metrics['avg_co2_reduction']} %", styles["Normal"]))
    elements.append(Paragraph(f"Average Cost Savings: ₹ {metrics['avg_cost_savings']}", styles["Normal"]))
    elements.append(Spacer(1, 20))

    # Table Data
    table_data = [["Material", "Predicted Cost", "Predicted CO2", "Score"]]

    for r in records:
        table_data.append([
            r.material_name,
            round(r.predicted_cost, 2),
            round(r.predicted_co2, 2),
            round(r.suitability_score, 4)
        ])

    table = Table(table_data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "CENTER")
    ]))

    elements.append(table)

    doc.build(elements)

    buffer.seek(0)

    return send_file(
        buffer,
        download_name="EcoPackAI_Summary_Report.pdf",
        as_attachment=True
    )



if __name__ == "__main__":
    app.run(debug=False)
