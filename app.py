from flask import Flask, request, render_template
import joblib
import numpy as np

from secom_failure_detection import logger
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
logger.info("Loading trained model and scaler...")
model_path = os.path.join("models", "model.pkl")
scaler_path = os.path.join("models", "scaler.pkl")

if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logger.info("Model and scaler loaded successfully.")
else:
    logger.error("Model or scaler file not found!")
    raise FileNotFoundError("Model or scaler file is missing!")


@app.route("/")
def index():
    logger.info("Serving home page...")
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            input_data = request.form.get("sensor_data").split(",")
            logger.info(f"Received input data: {input_data}")

            # Clean input data by stripping extra spaces
            input_data = [x.strip() for x in input_data]

            # Check if input data has the correct number of features
            # expected_features = 156
            # if len(input_data) != expected_features:
            #     logger.error(
            #         f"Input data does not have the correct number of features. Expected {expected_features}, got {len(input_data)}."
            #     )
            #     return render_template(
            #         "index.html",
            #         prediction_text=f"Error: Invalid number of features in input data. Expected {expected_features} features, but got {len(input_data)}.",
            #     )

            # Convert input data to a numpy array of floats
            input_data = np.array([input_data], dtype=float)

            # Scale the input data
            # input_data_scaled = scaler.transform(input_data)

            # Make prediction
            prediction = model.predict(input_data)

            # Map prediction to a readable result
            result = "Failure" if prediction[0] == 1 else "No Failure"
            logger.info(f"Prediction result: {result}")

            return render_template(
                "index.html", prediction_text=f"Prediction: {result}"
            )

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return render_template(
                "index.html",
                prediction_text="Error during prediction. Please check input data format.",
            )


if __name__ == "__main__":
   app.run(host='0.0.0.0', port=5000)

