# 🏥 Medical Cost Prediction System (ML + Flask Web App)

A Machine Learning web application that predicts medical insurance costs based on personal health, lifestyle, and medical attributes.

## 🧠 Tech Stack

- 🧠 Machine Learning Model (trained & optimized)
- ⚙️ Flask Backend
- 🎨 HTML/CSS Frontend
- 📊 Scikit-Learn / XGBoost

---

## 📌 Project Overview

This system predicts a person's estimated medical charges using health-related information such as age, BMI, smoking status, cholesterol, glucose, and lifestyle habits.

Multiple regression models were tested, and the best-performing model (**Random Forest Regressor**) was selected for final deployment.

---

## 📊 Dataset Details

- **Dataset:** `medical_cost_dataset.csv`
- **Source:** Kaggle
- **Target Variable:** `charges_usd`

### 🔑 Input Features

- Age
- Sex
- BMI
- Children
- Smoker
- Blood Pressure
- Cholesterol
- Glucose
- Heart Rate
- Exercise Level
- Alcohol Intake

---

## 🔧 Data Preprocessing

- Checked missing values
- Removed duplicates
- Encoded categorical columns:
  - Sex
  - Smoker
  - Exercise Level
  - Alcohol Intake
- Feature Scaling applied using **StandardScaler**
- Train-Test Split performed
- Prepared final dataset for model training

---

## 🧠 Models Tested

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor ✅ (Best)
- XGBoost Regressor

---

## 📈 Evaluation Metrics

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score
- Cross-validation Score

---

## 🏆 Final Model

**Random Forest Regressor**

```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
