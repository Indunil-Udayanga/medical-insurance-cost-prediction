from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model + preprocessing artifacts
model = pickle.load(open(os.path.join(BASE_DIR, "insurance_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
columns = pickle.load(open(os.path.join(BASE_DIR, "columns.pkl"), "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        # 1. Get form data
        data = {
            "age": int(request.form['age']),
            "bmi": float(request.form['bmi']),
            "children": int(request.form['children']),
            "blood_pressure": float(request.form['blood_pressure']),
            "cholesterol": float(request.form['cholesterol']),
            "glucose": float(request.form['glucose']),
            "heart_rate": float(request.form['heart_rate']),

            "sex": request.form['sex'],
            "smoker": request.form['smoker'],
            "exercise_level": request.form['exercise_level'],
            "alcohol_intake": request.form['alcohol_intake']
        }

        # 2. Create empty dataframe with correct columns
        df = pd.DataFrame(0, index=[0], columns=columns)

        # 3. Fill numeric values
        df["age"] = data["age"]
        df["bmi"] = data["bmi"]
        df["children"] = data["children"]
        df["blood_pressure"] = data["blood_pressure"]
        df["cholesterol"] = data["cholesterol"]
        df["glucose"] = data["glucose"]
        df["heart_rate"] = data["heart_rate"]

        # 4. Encode categorical features safely

        if "sex_male" in df.columns:
            df["sex_male"] = 1 if data["sex"] == "male" else 0

        if "smoker_yes" in df.columns:
            df["smoker_yes"] = 1 if data["smoker"] == "yes" else 0

        if "exercise_level_medium" in df.columns:
            df["exercise_level_medium"] = 1 if data["exercise_level"] == "medium" else 0

        if "exercise_level_high" in df.columns:
            df["exercise_level_high"] = 1 if data["exercise_level"] == "high" else 0

        if "alcohol_intake_moderate" in df.columns:
            df["alcohol_intake_moderate"] = 1 if data["alcohol_intake"] == "moderate" else 0

        if "alcohol_intake_high" in df.columns:
            df["alcohol_intake_high"] = 1 if data["alcohol_intake"] == "high" else 0

        # 5. Scale + Predict
        scaled = scaler.transform(df)
        prediction = model.predict(scaled)[0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted Insurance Cost: ${prediction:.2f}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)