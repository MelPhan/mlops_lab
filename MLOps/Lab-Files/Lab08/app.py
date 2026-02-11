from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
# Load the model
with open("wine_quality_model.pkl", "rb") as f:
    model = pickle.load(f)
# Initialize Flask app
app = Flask(__name__)
@app.route('/')
def home():
    return render_template("index.html")
@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from form
    data = [float(request.form[feature]) for feature in request.form.keys()]
    prediction = model.predict([data])
    return render_template("index.html", prediction_text=f"Predicted Wine Quality: {prediction[0]:.2f}")
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)