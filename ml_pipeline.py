import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# Load Iris dataset
data = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
# Display the first few rows
print(data.head())

data.hist(figsize=(8, 6))
plt.show()

# Summary of the dataset
print(data.info())
print(data.describe())
print("Missing values in the dataset:\n", data.isnull().sum())

import seaborn as sns
sns.pairplot(data, hue='species', markers=["o", "s", "D"])
plt.show()

sns.boxplot(x='species', y='sepal_length', data=data)
plt.title('Sepal Length Distribution by Species')
plt.show()

sns.violinplot(x='species', y='petal_length', data=data)
plt.title('Petal Length Distribution by Species')
plt.show()

# Create a new feature: sepal_area
data['sepal_area'] = data['sepal_length'] * data['sepal_width']
# Create a new feature: petal_area
data['petal_area'] = data['petal_length'] * data['petal_width']
# Display the new features
print(data[['sepal_area', 'petal_area']].head())

from sklearn.preprocessing import StandardScaler
# Select numerical columns
numerical_cols = ['sepal_length', 'sepal_width', 'petal_length',
'petal_width', 'sepal_area', 'petal_area']
# Apply Standard Scaling
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
# Display scaled features
print(data[numerical_cols].head())

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# Split data into features and target
X = data.drop('species', axis=1)
y = data['species']
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=42)
# Build an ML pipeline with preprocessing and model training steps
pipeline = Pipeline(steps=[
 ('classifier', RandomForestClassifier(n_estimators=100,
random_state=42))
])
# Train the pipeline
pipeline.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report
# Predict on test set
y_pred = pipeline.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
# Classification report
print(classification_report(y_test, y_pred))
