from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.joblib")
target_names = joblib.load("target_names.joblib")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided."}), 400
        
    features = data.get("features")
    
    if features is None or len(features) != 4 or not all(isinstance(x, (int, float)) for x in features):
        return jsonify({"error": "Invalid input. Send 4 numeric features."}), 400

    feat_array = np.array([features])
    pred_idx = model.predict(feat_array)[0]
    probs = model.predict_proba(feat_array)[0]

    return jsonify({
        "predicted_class": target_names[pred_idx],
        "probabilities": dict(zip(target_names.tolist(), np.round(probs, 4).tolist()))
    })

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided."}), 400
        
    samples = data.get("samples", [])
    
    if not isinstance(samples, list) or not samples:
        return jsonify({"error": "Send a list of samples."}), 400

    try:
        samples_array = np.array(samples)
        preds = model.predict(samples_array)
        probs = model.predict_proba(samples_array)
        
        results = []
        for s, p, prob in zip(samples, preds, probs):
            results.append({
                "features": s,
                "predicted_class": target_names[p],
                "probabilities": dict(zip(target_names.tolist(), np.round(prob, 4).tolist()))
            })
        
        return jsonify({"predictions": results})
    
    except Exception as e:
        return jsonify({"error": f"Error processing batch: {str(e)}"}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)