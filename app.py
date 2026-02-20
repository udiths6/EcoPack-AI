from flask import Flask, request, jsonify, send_file, make_response, render_template
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import base64
import os
import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sqlalchemy import create_engine
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------
# 1️⃣ CONFIGURATION
# ---------------------------------------------------

load_dotenv()

app = Flask(__name__)
CORS(app)

# Get database URL from environment variable
database_url = os.environ.get('DATABASE_URL')

# Fix for Render (they use postgres:// but SQLAlchemy needs postgresql://)
if database_url and database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)

app.config["SQLALCHEMY_DATABASE_URI"] = database_url or \
    "postgresql://postgres:2025@localhost:5432/ecopackai"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# Database Model
class Recommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_category = db.Column(db.String(50))
    fragility = db.Column(db.String(20))
    shipping_type = db.Column(db.String(20))
    sustainability_priority = db.Column(db.String(20))
    material_name = db.Column(db.String(100))
    predicted_cost = db.Column(db.Float)
    predicted_co2 = db.Column(db.Float)
    suitability_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

# Create tables on application startup (works with gunicorn)
try:
    with app.app_context():
        db.create_all()
        print("✅ Database tables initialized!")
except Exception as e:
    print(f"⚠️  Note: Table creation will be attempted again on startup: {e}")

# Load materials and models
df_materials = pd.read_csv("ecopackai_frozen_materials.csv")
co2_model = joblib.load("models/co2_model.pkl")
cost_model = joblib.load("models/cost_model.pkl")

# Baseline values for comparison (average of all materials)
BASELINE_CO2 = df_materials['co2_score'].mean()  # ~4.14
BASELINE_COST = df_materials['cost'].mean()  # ~4.96


@app.route("/", methods=["GET"])
def home():
    """Serve the main recommendation page"""
    return render_template('simple_ui.html')

@app.route("/dashboard", methods=["GET"])
def dashboard_page():
    """Serve the analytics dashboard"""
    return render_template('dashboard.html')


@app.route("/api/dashboard/analytics", methods=["GET"])
def get_dashboard_analytics():
    """
    Get comprehensive analytics for the dashboard
    Returns: Material usage, CO2 reduction, cost savings, trends
    """
    try:
        # Get all recommendations from database
        recommendations = Recommendation.query.all()
        
        if not recommendations:
            return jsonify({
                "status": "success",
                "message": "No data available yet. Make some recommendations first!",
                "data": None
            })
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([{
            'material_name': r.material_name,
            'predicted_cost': r.predicted_cost,
            'predicted_co2': r.predicted_co2,
            'suitability_score': r.suitability_score,
            'product_category': r.product_category,
            'sustainability_priority': r.sustainability_priority,
            'created_at': r.created_at
        } for r in recommendations])
        
     
        # 1. MATERIAL USAGE COUNT

        material_usage = df['material_name'].value_counts().head(10).to_dict()
        
        
        # 2. CO2 REDUCTION PERCENTAGE
        
        avg_predicted_co2 = df['predicted_co2'].mean()
        co2_reduction_percent = ((BASELINE_CO2 - avg_predicted_co2) / BASELINE_CO2) * 100
        total_co2_saved = (BASELINE_CO2 - avg_predicted_co2) * len(df)
        
        
        # 3. COST SAVINGS
        
        avg_predicted_cost = df['predicted_cost'].mean()
        cost_savings_percent = ((BASELINE_COST - avg_predicted_cost) / BASELINE_COST) * 100
        total_cost_saved = (BASELINE_COST - avg_predicted_cost) * len(df)
        
        
        # 4. ECO-FRIENDLY MATERIALS DISTRIBUTION
        
        # Merge with materials data to get biodegradability
        df_with_specs = df.merge(
            df_materials[['material_name', 'biodegradibility_score', 'recyclability_percentage']], 
            on='material_name', 
            how='left'
        )
        
        eco_friendly_count = len(df_with_specs[df_with_specs['biodegradibility_score'] >= 7])
        non_eco_count = len(df_with_specs[df_with_specs['biodegradibility_score'] < 7])
        
        
        # 5. TRENDS OVER TIME
        
        df['date'] = pd.to_datetime(df['created_at']).dt.date
        daily_trends = df.groupby('date').agg({
            'predicted_co2': 'mean',
            'predicted_cost': 'mean',
            'material_name': 'count'
        }).reset_index()
        
        daily_trends.columns = ['date', 'avg_co2', 'avg_cost', 'recommendation_count']
        daily_trends['date'] = daily_trends['date'].astype(str)
        
        
        # 6. CATEGORY BREAKDOWN
        
        category_breakdown = df.groupby('product_category').agg({
            'predicted_co2': 'mean',
            'predicted_cost': 'mean',
            'material_name': 'count'
        }).round(2).to_dict()
        
        
        # 7. TOP PERFORMING MATERIALS
        
        top_materials = df.groupby('material_name').agg({
            'suitability_score': 'mean',
            'predicted_co2': 'mean',
            'predicted_cost': 'mean',
            'material_name': 'count'
        }).round(3)
        top_materials.columns = ['avg_suitability', 'avg_co2', 'avg_cost', 'times_recommended']
        top_materials = top_materials.sort_values('avg_suitability', ascending=False).head(10)
        
        # Prepare response
        analytics_data = {
            "summary_cards": {
                "total_recommendations": len(df),
                "unique_materials_used": df['material_name'].nunique(),
                "co2_reduction_percent": round(co2_reduction_percent, 2),
                "total_co2_saved": round(total_co2_saved, 2),
                "cost_savings_percent": round(cost_savings_percent, 2),
                "total_cost_saved": round(total_cost_saved, 2),
                "avg_suitability_score": round(df['suitability_score'].mean() * 100, 1),
                "eco_friendly_percentage": round((eco_friendly_count / len(df)) * 100, 1)
            },
            "material_usage": material_usage,
            "eco_distribution": {
                "eco_friendly": eco_friendly_count,
                "non_eco_friendly": non_eco_count
            },
            "trends": daily_trends.to_dict('records'),
            "category_breakdown": category_breakdown,
            "top_materials": top_materials.reset_index().to_dict('records')
        }
        
        return jsonify({
            "status": "success",
            "data": analytics_data
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/api/dashboard/charts", methods=["GET"])
def get_dashboard_charts():
    """
    Generate interactive Plotly charts
    Returns: JSON data for charts
    """
    try:
        recommendations = Recommendation.query.all()
        
        if not recommendations:
            return jsonify({
                "status": "error",
                "message": "No data available"
            }), 404
        
        df = pd.DataFrame([{
            'material_name': r.material_name,
            'predicted_cost': r.predicted_cost,
            'predicted_co2': r.predicted_co2,
            'suitability_score': r.suitability_score,
            'product_category': r.product_category,
            'created_at': r.created_at
        } for r in recommendations])
        
        # Merge with material specs
        df_with_specs = df.merge(
            df_materials[['material_name', 'biodegradibility_score', 'recyclability_percentage']], 
            on='material_name', 
            how='left'
        )
        
        charts_data = {}
        
        
        # CHART 1: Material Usage Bar Chart
        
        material_counts = df['material_name'].value_counts().head(10)
        charts_data['material_usage'] = {
            'labels': material_counts.index.tolist(),
            'values': material_counts.values.tolist()
        }
        
        
        # CHART 2: Eco-Friendly Distribution
        
        eco_counts = df_with_specs['biodegradibility_score'].apply(
            lambda x: 'Eco-Friendly (≥7)' if x >= 7 else 'Non-Eco (<7)'
        ).value_counts()
        charts_data['eco_distribution'] = {
            'labels': eco_counts.index.tolist(),
            'values': eco_counts.values.tolist()
        }
        
        
        # CHART 3: CO2 Reduction Trend
        
        df['date'] = pd.to_datetime(df['created_at']).dt.date
        daily_co2 = df.groupby('date')['predicted_co2'].mean().reset_index()
        daily_co2['co2_reduction'] = ((BASELINE_CO2 - daily_co2['predicted_co2']) / BASELINE_CO2) * 100
        charts_data['co2_trend'] = {
            'dates': [str(d) for d in daily_co2['date'].tolist()],
            'values': daily_co2['co2_reduction'].tolist()
        }
        
        
        # CHART 4: Cost Savings Trend
        
        daily_cost = df.groupby('date')['predicted_cost'].mean().reset_index()
        daily_cost['cost_savings'] = BASELINE_COST - daily_cost['predicted_cost']
        charts_data['cost_trend'] = {
            'dates': [str(d) for d in daily_cost['date'].tolist()],
            'values': daily_cost['cost_savings'].tolist()
        }
        
        
        # CHART 5: Top Materials Ranking
        
        top_materials = df.groupby('material_name').agg({
            'suitability_score': 'mean'
        }).sort_values('suitability_score', ascending=True).tail(10)
        charts_data['top_materials'] = {
            'labels': top_materials.index.tolist(),
            'values': (top_materials['suitability_score'] * 100).tolist()
        }
        
        
        # CHART 6: Category Comparison
        
        category_stats = df.groupby('product_category').agg({
            'predicted_co2': 'mean',
            'predicted_cost': 'mean'
        }).reset_index()
        charts_data['category_comparison'] = {
            'categories': category_stats['product_category'].tolist(),
            'co2_values': category_stats['predicted_co2'].round(2).tolist(),
            'cost_values': category_stats['predicted_cost'].round(2).tolist()
        }
        
        return jsonify({
            "status": "success",
            "data": charts_data
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/api/export/pdf", methods=["GET"])
def export_pdf_report():
    """
    Export sustainability report as PDF
    """
    try:
        recommendations = Recommendation.query.all()
        
        if not recommendations:
            return jsonify({"status": "error", "message": "No data to export"}), 404
        
        df = pd.DataFrame([{
            'material_name': r.material_name,
            'predicted_cost': r.predicted_cost,
            'predicted_co2': r.predicted_co2,
            'suitability_score': r.suitability_score,
            'product_category': r.product_category,
            'created_at': r.created_at
        } for r in recommendations])
        
        # Calculate metrics
        avg_co2 = df['predicted_co2'].mean()
        co2_reduction = ((BASELINE_CO2 - avg_co2) / BASELINE_CO2) * 100
        avg_cost = df['predicted_cost'].mean()
        cost_savings = BASELINE_COST - avg_cost
        
        # Create PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2ecc71'),
            spaceAfter=30,
            alignment=1  # Center
        )
        elements.append(Paragraph("EcoPack AI Sustainability Report", title_style))
        elements.append(Spacer(1, 20))
        
        # Report date
        date_text = f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        elements.append(Paragraph(date_text, styles['Normal']))
        elements.append(Spacer(1, 30))
        
        # Summary Section
        elements.append(Paragraph("Executive Summary", styles['Heading2']))
        elements.append(Spacer(1, 10))
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Recommendations', str(len(df))],
            ['Unique Materials Used', str(df['material_name'].nunique())],
            ['Average CO2 Emissions', f"{avg_co2:.2f}"],
            ['CO2 Reduction vs Baseline', f"{co2_reduction:.2f}%"],
            ['Average Cost', f"${avg_cost:.2f}"],
            ['Cost Savings vs Baseline', f"${cost_savings:.2f}"],
            ['Average Suitability Score', f"{df['suitability_score'].mean()*100:.1f}%"],
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ecc71')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 30))
        
        # Top Materials
        elements.append(Paragraph("Top Recommended Materials", styles['Heading2']))
        elements.append(Spacer(1, 10))
        
        top_materials = df['material_name'].value_counts().head(10)
        materials_data = [['Rank', 'Material', 'Times Recommended']]
        for idx, (material, count) in enumerate(top_materials.items(), 1):
            materials_data.append([str(idx), material, str(count)])
        
        materials_table = Table(materials_data, colWidths=[0.75*inch, 3*inch, 1.5*inch])
        materials_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(materials_table)
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f'sustainability_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
            mimetype='application/pdf'
        )
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/export/excel", methods=["GET"])
def export_excel_report():
    """
    Export full ranking table as Excel
    """
    try:
        recommendations = Recommendation.query.all()
        
        if not recommendations:
            return jsonify({"status": "error", "message": "No data to export"}), 404
        
        df = pd.DataFrame([{
            'ID': r.id,
            'Material Name': r.material_name,
            'Product Category': r.product_category,
            'Fragility': r.fragility,
            'Shipping Type': r.shipping_type,
            'Sustainability Priority': r.sustainability_priority,
            'Predicted Cost': r.predicted_cost,
            'Predicted CO2': r.predicted_co2,
            'Suitability Score': r.suitability_score,
            'Recommended At': r.created_at
        } for r in recommendations])
        
        # Create Excel file in memory
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Full data sheet
            df.to_excel(writer, sheet_name='All Recommendations', index=False)
            
            # Summary sheet
            summary_df = pd.DataFrame({
                'Metric': [
                    'Total Recommendations',
                    'Unique Materials',
                    'Average CO2',
                    'Average Cost',
                    'Average Suitability Score'
                ],
                'Value': [
                    len(df),
                    df['Material Name'].nunique(),
                    f"{df['Predicted CO2'].mean():.2f}",
                    f"${df['Predicted Cost'].mean():.2f}",
                    f"{df['Suitability Score'].mean()*100:.1f}%"
                ]
            })
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Top materials sheet
            top_materials = df['Material Name'].value_counts().reset_index()
            top_materials.columns = ['Material', 'Count']
            top_materials.to_excel(writer, sheet_name='Top Materials', index=False)
        
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f'recommendations_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# api route

@app.route("/api", methods=["POST"])
def material():
    """Material recommendation API endpoint"""
    
    try:
        # Load full dataset
        df = df_materials.copy()
        
        # Get user inputs
        data = request.get_json()
        prod_cat = data["Product_category"].lower()
        fragility = data["Fragility"].lower()
        ship_type = data["Shipping_type"].lower()
        sust_prio = data["Sustainability_priority"].lower()
        
        # Apply filtering
        if fragility == "high":
            df = df[df["strength"] >= 3]
        elif fragility == "medium":
            df = df[df["strength"] >= 2]

        if prod_cat == "food":
            df = df[df["biodegradibility_score"] >= 7]
        elif prod_cat == "electronics":
            df = df[df["strength"] >= 2]

        if ship_type == "international":
            df = df[df["strength"] >= 2]
        
        if df.empty:
            return jsonify({
                "status": "fail",
                "message": "No suitable materials found for the given constraints"
            }), 404
        
        # Make predictions
        features = ["strength", "weight_capacity", "biodegradibility_score", "recyclability_percentage"]
        x = df[features]

        scaler = joblib.load("models/feature_scaler.pkl")
        x_scaled = scaler.transform(x)
        
        df["predicted_cost"] = cost_model.predict(x_scaled)
        df["predicted_co2"] = co2_model.predict(x_scaled)
        
        # Normalization
        df["cost_norm"] = 1 - MinMaxScaler().fit_transform(df[["predicted_cost"]]).flatten()
        df["co2_norm"] = 1 - MinMaxScaler().fit_transform(df[["predicted_co2"]]).flatten()
        df["strength_norm"] = MinMaxScaler().fit_transform(df[["strength"]]).flatten()
        
        # Weight management
        eco_weight = 0.4
        cost_weight = 0.4
        strength_weight = 0.2
        
        if sust_prio == "high":
            eco_weight += 0.3
            cost_weight -= 0.3
        elif sust_prio == "medium":
            eco_weight += 0.15
            cost_weight -= 0.15
        elif sust_prio == "low":
            eco_weight -= 0.20
            cost_weight += 0.20

        if ship_type == "international":
            eco_weight += 0.1
            strength_weight += 0.1
        
        # Normalize weights
        total = eco_weight + cost_weight + strength_weight
        eco_weight /= total
        cost_weight /= total
        strength_weight /= total
        
        # Calculate suitability score
        df["suitability_score"] = (
            eco_weight * df["co2_norm"] +
            cost_weight * df["cost_norm"] +
            strength_weight * df["strength_norm"]
        )
        
        df = df.sort_values("suitability_score", ascending=False)
        
        top_df = df.head(3).reset_index(drop=True)
        top_df["rank"] = top_df.index + 1

        # Save to database
        for _, row in top_df.iterrows():
            rec = Recommendation(
                product_category=prod_cat,
                fragility=fragility,
                shipping_type=ship_type,
                sustainability_priority=sust_prio,
                material_name=row["material_name"],
                predicted_cost=float(row["predicted_cost"]),
                predicted_co2=float(row["predicted_co2"]),
                suitability_score=float(row["suitability_score"])
            )
            db.session.add(rec)

        db.session.commit()
        
        # Return response
        response = {
            "status": "success",
            "recommended_materials": top_df[[
                "rank",
                "material_name",
                "predicted_cost",
                "predicted_co2",
                "suitability_score"
            ]].to_dict(orient="records")
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    # Create database tables on startup
    with app.app_context():
        try:
            db.create_all()
            print("✅ Database tables created successfully!")
            
            # Verify tables exist
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            print(f"📋 Available tables: {tables}")
            
            if 'recommendation' in tables:
                print("✅ 'recommendation' table is ready!")
            else:
                print("⚠️  Warning: 'recommendation' table not found!")
                
        except Exception as e:
            print(f"❌ Error creating tables: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the app
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
# ---------------------------------------------------
# 2️⃣ DATABASE CONFIG (PRODUCTION SAFE)
# ---------------------------------------------------

DB_URI = os.getenv("DATABASE_URL")

if not DB_URI:
    raise Exception("DATABASE_URL environment variable not set")

# Fix Render postgres:// issue
if DB_URI.startswith("postgres://"):
    DB_URI = DB_URI.replace("postgres://", "postgresql://", 1)

engine = create_engine(DB_URI)

# ---------------------------------------------------
# 3️⃣ MODEL LOADING
# ---------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

def load_model(filename):
    path = BASE_DIR / filename
    if not path.exists():
        print(f"⚠ {filename} not found")
        return None
    return joblib.load(path)

rf_model = load_model("rf_model.pkl")
xgb_model = load_model("xgb_model.pkl")
scaler = load_model("scaler.pkl")

# ---------------------------------------------------
# 4️⃣ DATA UTILITIES
# ---------------------------------------------------

def fetch_data():
    return pd.read_sql("SELECT * FROM materials", engine)

def safe_normalize(value, min_val, max_val):
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)

# ---------------------------------------------------
# 5️⃣ CATEGORY RULES
# ---------------------------------------------------

Category_rules = {
    "food": lambda df: df[df["biodegradability_score"] >= 8],
    "beverages": lambda df: df[(df["strength"] >= 3) & (df["recyclability"] >= 70)],
    "pharmaceuticals": lambda df: df[df["biodegradability_score"] >= 6],
    "agriculture": lambda df: df[df["biodegradability_score"] >= 9],
    "electronics": lambda df: df[df["strength"] >= 4],
    "automotive_parts": lambda df: df[df["strength"] >= 5],
    "construction_tools": lambda df: df[df["weight_capacity"] >= 50],
    "industrial_chemicals": lambda df: df[(df["strength"] >= 5) & (df["recyclability"] >= 50)],
    "cosmetics": lambda df: df[df["recyclability"] >= 80],
    "apparel_fashion": lambda df: df[df["biodegradability_score"] >= 7],
    "luxury_goods": lambda df: df[df["cost_per_unit"] >= 100],
    "e_commerce_general": lambda df: df[df["recyclability"] >= 60],
    "home_appliances": lambda df: df[df["strength"] >= 4],
    "toys_baby_products": lambda df: df[(df["biodegradability_score"] >= 8) & (df["strength"] >= 2)],
    "office_supplies": lambda df: df[df["recyclability"] >= 90]
}

# ---------------------------------------------------
# 6️⃣ ROUTES
# ---------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "online"})

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json(force=True)

        required_fields = [
            "product_category",
            "fragility",
            "Shipping_Type",
            "Sustainability_Priority"
        ]

        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        full_df = fetch_data()

        if full_df.empty:
            return jsonify({"error": "No materials available"}), 500

        # ---------------- CATEGORY FILTER
        category = data["product_category"].lower()

        if category in Category_rules:
            filtered = Category_rules[category](full_df)
            materials_df = filtered if not filtered.empty else full_df.copy()
        else:
            materials_df = full_df.copy()

        # ---------------- FRAGILITY FILTER
        fragility = data["fragility"].lower()

        if fragility == "high":
            strength_boost = 0.2
            temp = materials_df[materials_df["strength"] >= 4]
        elif fragility == "medium":
            strength_boost = 0.1
            temp = materials_df[materials_df["strength"] >= 2]
        else:
            strength_boost = 0.0
            temp = materials_df

        if not temp.empty:
            materials_df = temp

        # ---------------- SHIPPING FILTER
        shipping = data["Shipping_Type"].lower()

        if shipping == "international":
            temp = materials_df[materials_df["strength"] >= 3]
            if not temp.empty:
                materials_df = temp

        # ---------------- PRIORITY WEIGHTS
        priority = data["Sustainability_Priority"].lower()

        if priority == "high":
            w_cost, w_co2, w_suit = 0.10, 0.55, 0.35
        elif priority == "medium":
            w_cost, w_co2, w_suit = 0.30, 0.40, 0.30
        else:
            w_cost, w_co2, w_suit = 0.60, 0.25, 0.15

        # ---------------- ML CO2 PREDICTION
        if scaler and xgb_model:
            rename_map = {
                "strength": "Strength",
                "weight_capacity": "Weight_Capacity",
                "cost_per_unit": "Cost_Per_Unit_INR",
                "biodegradability_score": "Biodegradability_Score",
                "recyclability": "Recyclability"
            }

            feature_order = list(rename_map.values())

            X_input = materials_df[list(rename_map.keys())].rename(columns=rename_map)
            X_input = X_input[feature_order]
            X_scaled = scaler.transform(X_input)
            co2_preds = xgb_model.predict(X_scaled)
        else:
            co2_preds = materials_df["co2_emission_score"].values

        # ---------------- NORMALIZATION RANGES
        min_cost = full_df["cost_per_unit"].min()
        max_cost = full_df["cost_per_unit"].max()
        min_co2 = max(0.0, full_df["co2_emission_score"].min())
        max_co2 = full_df["co2_emission_score"].max()

        results = []

        for i, (_, row) in enumerate(materials_df.iterrows()):

            cost = max(0.0, float(row["cost_per_unit"]))
            pred_co2 = max(0.0, float(co2_preds[i]))  # NEVER NEGATIVE

            s_norm = safe_normalize(row["strength"], full_df["strength"].min(), full_df["strength"].max())
            r_norm = safe_normalize(row["recyclability"], full_df["recyclability"].min(), full_df["recyclability"].max())
            b_norm = safe_normalize(row["biodegradability_score"], full_df["biodegradability_score"].min(), full_df["biodegradability_score"].max())
            cost_norm = safe_normalize(cost, min_cost, max_cost)
            co2_norm = safe_normalize(pred_co2, min_co2, max_co2)

            sustainability = (
                (0.4 + strength_boost) * s_norm +
                0.3 * r_norm +
                0.3 * b_norm
            )

            final_score = (
                w_cost * (1 - cost_norm) +
                w_co2 * (1 - co2_norm) +
                w_suit * sustainability
            )

            final_score = max(0.0, final_score)

            results.append({
                "material": row["material_type"],
                "predicted_cost": round(cost, 2),
                "predicted_co2": round(pred_co2, 2),
                "suitability_score": round(final_score * 100, 2)
            })

        df_results = pd.DataFrame(results)
        top5 = df_results.sort_values("suitability_score", ascending=False).head(5)

        return jsonify({
            "recommended_materials": top5.to_dict(orient="records")
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": "Internal Server Error"}), 500


# ---------------------------------------------------
# 7️⃣ ENTRY POINT (FOR RENDER)
# ---------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
