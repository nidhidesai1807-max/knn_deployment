from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Load model paths correctly (important for Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "knn_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

app = Flask(__name__)

# Home route
@app.route("/")
def home():
    return "KNN Model API is running!"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features)
        
        return jsonify({"prediction": int(prediction[0])})
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Important for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)