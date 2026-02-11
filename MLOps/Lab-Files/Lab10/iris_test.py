import time
import random
import numpy as np
from prometheus_client import Counter, Summary, Gauge, start_http_server
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# Define Prometheus metrics
REQUEST_COUNT = Counter('request_count', 'Total number of requests to the model')
INFERENCE_TIME = Summary('inference_time', 'Time spent processing predictions')
PREDICTION_ACCURACY = Gauge('prediction_accuracy', 'Model prediction accuracy (if ground truth is available)')

# Load the trained model
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the Iris dataset for synthetic data generation
iris = load_iris()
X, y = iris.data, iris.target

# Start the Prometheus metrics server
start_http_server(8001)

print("Prometheus metrics server running at http://localhost:8001/metrics")

# Function to periodically call the model and record metrics
def simulate_model_requests():

    while True:
        # Increment request count
        REQUEST_COUNT.inc()
       
        # Simulate input data
        input_data = random.choice(X)
        ground_truth = y[np.where(X == input_data)[0][0]]
       
        # Start timing the inference
        start_time = time.time()
        prediction = model.predict([input_data])
        inference_time = time.time() - start_time
       
        # Record inference time
        INFERENCE_TIME.observe(inference_time)
        print(f"Inference Time: {inference_time:.4f} seconds")
        # Evaluate prediction accuracy (if ground truth is available)

        is_correct = int(prediction[0] == ground_truth)
        PREDICTION_ACCURACY.set(is_correct)
        print(f"Prediction Accuracy: {is_correct}")

        # Wait before the next simulation (e.g., 5 seconds)
        time.sleep(5)
# Start simulating model requests
simulate_model_requests()
