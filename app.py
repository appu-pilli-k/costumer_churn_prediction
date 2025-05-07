from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load('../models/churn_model.pkl')
# Root endpoint
@app.route('/')
def index():
    return "✅ Customer Churn Prediction API is running. Use POST /predict with JSON input."

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert data to DataFrame
        input_df = pd.DataFrame([data])

        # Predict using model
        prediction = model.predict(input_df)

        # Return the prediction as JSON
        return jsonify({'churn_prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
