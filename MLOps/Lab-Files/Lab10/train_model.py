from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle
# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
# Train a Random Forest model
model = RandomForestClassifier()
model.fit(X, y)
# Save the model
with open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model trained and saved as iris_model.pkl")
