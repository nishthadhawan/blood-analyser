from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("jupyter notebook/cardio_model.pkl")

# Features expected by the model
FEATURES = [
    "Cholesterol",
    "HDL Cholesterol",
    "LDL Cholesterol",
    "Triglycerides",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Heart Rate",
    "BMI",
    "C-reactive Protein"
]


@app.route("/")
def home():
    return jsonify({
        "message": "Cardiovascular Disease Prediction API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        missing_features = [
            feature for feature in FEATURES
            if feature not in data
        ]

        if missing_features:
            return jsonify({
                "error": "Missing features",
                "missing": missing_features
            }), 400

        input_data = pd.DataFrame(
            [[data[feature] for feature in FEATURES]],
            columns=FEATURES
        )

        prediction = model.predict(input_data)

        return jsonify({
            "prediction": int(prediction[0])
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)