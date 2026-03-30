from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load BOTH models
rf_model = joblib.load("rf_model.pkl")
lr_model = joblib.load("lr_model.pkl")


@app.route("/predict", methods=["GET"])
def predict():
    try:
        # GET parameters (adjust based on your system inputs)
        year = float(request.args.get("year", 0))
        employed = float(request.args.get("employed", 0))
        unemployed = float(request.args.get("unemployed", 0))

        data = np.array([[year, employed, unemployed]])

        # Predict using both models
        rf_pred = rf_model.predict(data)[0]
        lr_pred = lr_model.predict(data)[0]

        return jsonify({
            "rf_prediction": float(rf_pred),
            "lr_prediction": float(lr_pred)
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run()
