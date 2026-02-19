🌱 EcoPack-AI

AI-Based Sustainable Packaging Recommendation System built using Flask, Machine Learning, and PostgreSQL.

## Project Overview

EcoPack-AI recommends the most suitable packaging material based on:

∙ Product category

∙ Fragility level

∙ Shipping type

∙ Sustainability priority

## The system predicts:

∙ Packaging Cost

∙ CO₂ Impact

∙ Material Suitability Score

Recommendations are ranked and stored in a PostgreSQL database, with dashboard analytics and export options.


## Tech Stack

∙ Python, Flask

∙ SQLAlchemy + PostgreSQL

∙ Scikit-Learn, XGBoost

∙ Pandas

∙ Plotly (Dashboard)

∙ ReportLab (PDF Export)

∙ Gunicorn (Production)


## Features

∙ Intelligent recommendation engine

∙ Dynamic weighted scoring logic

∙ Interactive sustainability dashboard

∙ Excel & PDF export

∙ Secure API endpoint with API key authentication


## Environment Variables

Required:

∙ API_KEY
∙ DATABASE_URL

▶ Run Locally
cd Backend
pip install -r requirements.txt
python app.py


## Deployment (Render)

Build Command:

∙ pip install -r Backend/requirements.txt


Start Command:

 ∙ gunicorn Backend.app:app
