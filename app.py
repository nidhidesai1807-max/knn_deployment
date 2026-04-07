import os
print("Current Working Directory:", os.getcwd())
print("Files in directory:", os.listdir())
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return "KNN Model API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    
    # Convert input to numpy array
    features = np.array(data).reshape(1, -1)
    
    # Scale input
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)
    
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)