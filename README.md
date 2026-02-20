
# 🌱 EcoPack AI

### Sustainable Packaging Recommendation System

EcoPack AI is a machine learning-based web application that recommends eco-friendly packaging materials by optimizing cost and CO₂ emissions based on product and shipping conditions.

---

# 🏗 1️⃣ Architecture Diagram

## System Architecture Overview

![architecture](images/architecture_new.png)

### Layer Description:

* **User Interface Layer** – Collects product inputs (category, fragility, shipping type, sustainability priority).
* **Application Layer (Flask)** – Processes API requests and connects frontend with ML models and database.
* **Machine Learning Layer** – Predicts packaging cost and CO₂ emissions, ranks materials.
* **Database Layer (PostgreSQL)** – Stores dataset and recommendation history.
* **Analytics Layer** – Displays insights, charts, and exportable reports.

---

# 📊 2️⃣ Dataset Description

The materials dataset was collected from Kaggle and HuggingFace and imported into PostgreSQL.

### Dataset Features:

* Material id
* Material name
* Strength
* Weight capacity
* Cost
* Biodegradibility score
* Co2score
* Recyclability percentage

### Data Preprocessing:

* Handled missing values
* Normalized numerical features
* Encoded categorical variables
* Feature selection for model training

---

# 🤖 3️⃣ Machine Learning Model Explanation

Two separate regression models were developed:

## 🔹 Random Forest (Cost Prediction)

* Used for predicting packaging cost
* Handles non-linear relationships
* Reduces overfitting through ensemble learning
* Performs well with structured tabular data

## 🔹 XGBoost (CO₂ Prediction & Ranking)

* Used to predict CO₂ footprint
* High accuracy using gradient boosting
* Efficient and scalable
* Handles complex feature interactions

### Ranking Logic

Final material ranking is based on:

* Predicted Cost
* Predicted CO₂ Emission
* Suitability Score
* Dynamic weighting based on sustainability priority

    Suitability Score ​= (Weco​×CO2norm​)+(Wcost​×Costnorm​)+(Wstrength​×Strengthnorm​)

Materials are filtered based on category, fragility, shipping type requirements before ranking.

---

# 📈 4️⃣ Evaluation Metrics

Model performance was evaluated using:

* **MAE (Mean Absolute Error)** – Measures average absolute prediction error
* **RMSE (Root Mean Squared Error)** – Penalizes larger errors
* **R² Score** – Measures how well the model explains variance

## 🔹 Random Forest

![Random Forest](images/rf.png)

## 🔹 XGBoost

![XGBoost](images/xgb.png)

Data was split into training and testing sets to ensure unbiased evaluation.

---

# 📊 5️⃣ Dashboard Explanation

The Business Intelligence Dashboard provides:

* Top 3 recommended materials
* Cost comparison charts
* CO₂ emission reduction analysis
* Material performance trends
* Exportable PDF/Excel reports

### Key Insights:

* Visual comparison of sustainability impact
* Cost efficiency tracking
* Historical recommendation analysis

The dashboard enables data-driven decision making for eco-friendly packaging selection.

---

# ☁️ 6️⃣ Deployment Steps

The project is deployed on Render with PostgreSQL integration.

### Deployment Process:

1. Push project to GitHub
2. Connect repository to Render
3. Configure environment variables:

   ```
   DATABASE_URL
   SECRET_KEY
   FLASK_ENV=production
   ```
4. Add `runtime.txt` for Python version
5. Install dependencies using:

   ```
   pip install -r requirements.txt
   ```
6. Ensure database tables are created using:

   ```python
   db.create_all()
   ```
7. Deploy and test APIs and dashboard functionality

---

# 🖼 7️⃣ Screenshots

## 🔹 Home Page

![Recommendation Page](images/recommendation.png)

## 🔹 Analytics Dashboard

![Dashboard](images/dashboard.png)

## 🔹 CO₂ & Cost Comparison Charts

![cost and co2 trend](images/costco2trend.png)

## 🔹 other Charts

![other charts](images/othercharts.png)

---

# 👨‍💻 Author

**Deb Sundar Adak**
AI intern

# 🌱 EcoPackAI -- Sustainable Material Recommendation & Cost Prediction System

EcoPackAI is a Machine Learning--powered web application that recommends
eco-friendly packaging materials based on cost, CO₂ score, and
suitability. It also predicts material cost using a trained Random
Forest model.

------------------------------------------------------------------------

## 🚀 Features

### ✅ 1. Material Recommendation System

-   Ranks materials based on:
    -   Cost
    -   CO₂ Score
    -   Suitability
-   Weighted scoring logic
-   Returns Top-N best materials

### ✅ 2. Cost Prediction Model

-   Trained using Random Forest Regressor
-   Predicts cost using:
    -   CO₂ Score
    -   Suitability
    -   Material features
-   Optimized using R², RMSE, and MAE metrics

### ✅ 3. REST API

-   JSON-based API endpoints
-   CORS enabled
-   Database integrated

### ✅ 4. PostgreSQL Integration

-   Stores materials dataset
-   Connected via SQLAlchemy

------------------------------------------------------------------------

## 🏗 Tech Stack

  Layer        Technology
  ------------ ------------------------------
  Backend      Flask
  ML Model     Scikit-learn (Random Forest)
  Database     PostgreSQL
  ORM          SQLAlchemy
  Frontend     HTML, CSS
  Deployment   Localhost (Flask Dev Server)

------------------------------------------------------------------------

## 📂 Project Structure

    EcoPackAI/
    │
    ├── app.py                 # Main Flask API
    ├── model.pkl              # Trained ML model
    ├── scaler.pkl             # Feature scaler
    ├── templates/
    │   └── index.html         # Frontend UI
    ├── static/                # CSS & JS
    ├── materials.csv          # Dataset
    ├── .env                   # Environment variables
    └── README.md

------------------------------------------------------------------------

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

``` bash
git clone https://github.com/your-username/EcoPackAI.git
cd EcoPackAI
```

### 2️⃣ Create Virtual Environment

``` bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies

``` bash
pip install -r requirements.txt
```

### 4️⃣ Setup PostgreSQL

-   Create database: `EcoPackAI`
-   Import materials table
-   Update `.env` file:

```{=html}
<!-- -->
```
    DATABASE_URL=postgresql://username:password@localhost:5432/EcoPackAI

### 5️⃣ Run the Application

``` bash
python app.py
```

Server will run on:

    http://127.0.0.1:5000

------------------------------------------------------------------------

## 🔌 API Endpoints

### 🔹 Health Check

    GET /

### 🔹 Get Recommendations

    POST /recommend

### 🔹 Predict Cost

    POST /predict

------------------------------------------------------------------------

## 🧠 Machine Learning Details

-   Model: Random Forest Regressor
-   Target Variable: Cost
-   Features:
    -   CO₂ Score
    -   Suitability
    -   Material characteristics
-   Evaluation Metrics:
    -   R² Score
    -   RMSE
    -   MAE

------------------------------------------------------------------------

## 📊 Recommendation Logic

1.  Normalize features\
2.  Apply weighted scoring\
3.  Rank materials\
4.  Return Top-N results

------------------------------------------------------------------------

## 📌 License

This project is for academic and learning purposes.
