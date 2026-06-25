# ChurnIQ — Customer Churn Intelligence and Retention Prediction System

## Business Intelligence + Machine Learning + Predictive Analytics

ChurnIQ is an end-to-end machine learning system designed to predict customer churn, analyze customer behavior, identify high-risk customers, and generate actionable retention strategies.

The project combines predictive analytics, business intelligence, explainable AI, and customer segmentation into a production-ready dashboard.

---

## Project Goal

The goal of ChurnIQ is to help businesses reduce customer churn by:

- Predicting customer churn probability
- Identifying high-risk customer groups
- Understanding churn-causing factors
- Generating retention strategies
- Estimating potential revenue loss
- Supporting data-driven business decisions

---

## Full Form

**ChurnIQ**  
**Customer Churn Intelligence and Retention Prediction System**

---

## Target Industries

- Telecom
- SaaS
- Subscription Businesses
- Banking
- Insurance
- E-commerce

---

## Tech Stack

### Core Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Plotly
- Joblib

### Explainability & Analytics

- SHAP
- Matplotlib
- Seaborn

### Machine Learning Models

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Support Vector Machine

---

## Core Features

### Data Processing

- Missing value handling
- Duplicate removal
- Feature encoding
- Feature scaling
- Outlier detection

### Exploratory Data Analysis

- Churn distribution analysis
- Monthly charges analysis
- Contract analysis
- Payment method analysis
- Customer tenure analysis
- Service usage analysis
- Correlation heatmap

### Model Training & Comparison

- Multiple model training
- Automatic best model selection
- Business-focused metric scoring

### Real-Time Prediction Dashboard

- Predict churn probability
- Risk categorization
- Business recommendations

### Explainable AI

- Feature importance visualization
- Top churn-causing factors

### Customer Segmentation

Customer segmentation using KMeans clustering:

- High Value Loyal
- Premium At Risk
- New / Low Value
- Moderate Value

### Retention Engine

Generates retention strategies:

- Loyalty rewards
- Discounts
- Plan optimization
- Personalized support

### Downloadable Reports

- Customer segmentation report
- Business analysis reports

---

## Project Structure

```text
ChurnIQ/
│── app.py
│── train.py
│── model.py
│── utils.py
│── shap_analysis.py
│── customer_segmentation.py
│── requirements.txt
│── README.md
│── .gitignore
│
├── dataset/
├── models/
├── reports/
├── screenshots/
├── notebooks/
```

---

## Machine Learning Workflow

```text
Dataset
↓
Data Cleaning
↓
EDA
↓
Feature Engineering
↓
Model Training
↓
Model Evaluation
↓
Best Model Selection
↓
Explainability
↓
Segmentation
↓
Deployment
```

---

## Model Evaluation Metrics

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report

---

## Best Model Performance

Current best model:

**Logistic Regression**

### Performance:

| Metric | Score |
|---|---|
| Accuracy | 73.81% |
| Precision | 50.43% |
| Recall | 78.34% |
| F1 Score | 61.36% |
| ROC-AUC | 84.12% |

---

## Why Logistic Regression Was Selected

Although XGBoost achieved higher accuracy, Logistic Regression was selected because:

- Higher recall (important for churn detection)
- Better business risk detection
- Lower false negatives
- Better interpretability

Business priority is identifying churners, not just maximizing accuracy.

---

## Screenshots

### Business Overview

![Business Overview](screenshots/business-overview.png)

---

### EDA Dashboard

![EDA Dashboard](screenshots/eda-dashboard.png)

---

### Model Performance

![Model Performance](screenshots/model-performance.png)

---

### Churn Prediction

![Prediction Dashboard](screenshots/prediction-dashboard.png)

---

### Retention Engine

![Retention Engine](screenshots/retention-engine.png)

---

### Feature Insights

![Feature Insights](screenshots/feature-insights.png)

---

### Explainable AI

![Explainable AI](screenshots/explainable-ai.png)

---

### Customer Segmentation

![Customer Segmentation](screenshots/customer-segmentation.png)

---

## Dataset Information

Dataset used:

IBM Telco Customer Churn Dataset

Features include:

- Customer tenure
- Monthly charges
- Total charges
- Contract type
- Payment method
- Internet services
- Technical support
- Online security
- Dependents

Target variable:

- Churn (Yes / No)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/build-with-saurav/ChurnIQ.git
cd ChurnIQ
```

Create virtual environment:

```bash
python3 -m venv churniq_env
source churniq_env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train model:

```bash
python train.py
```

Run Streamlit:

```bash
streamlit run app.py
```

---

## Business Insights

Key findings:

- Month-to-month customers churn the most
- Higher monthly charges increase churn probability
- New customers are at higher risk
- Electronic check users churn more
- Long-term contracts improve retention

---

## Future Improvements

- SHAP local explanations
- Churn trend forecasting
- Revenue forecasting
- Customer lifetime value prediction
- Automated email retention campaigns
- Deep learning models
- Cloud deployment

---

## Deployment

Deploy using Streamlit Community Cloud.

---

## GitHub Topics

```text
machine-learning
customer-churn-prediction
business-intelligence
predictive-analytics
streamlit
xgboost
classification
customer-retention
explainable-ai
shap
customer-segmentation
data-science
```

---

## Author

Saurav Kumar Singh

B.Tech CSE — NIT Calicut

AI | ML | Data Science | Predictive Analytics
