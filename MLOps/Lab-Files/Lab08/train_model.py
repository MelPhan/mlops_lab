import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
# Load dataset
data = pd.read_csv("winequality-red.csv")
X = data.drop("quality", axis=1)  # Features
y = data["quality"]  # Target variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
# Save the model
with open("wine_quality_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model trained and saved as wine_quality_model.pkl")