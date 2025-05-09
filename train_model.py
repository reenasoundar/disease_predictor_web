import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_excel("synthetic_symptoms_dataset_realistic_100_diseases.xlsx")

# Convert 'yes' to 1, 'no' to 0
X = df.drop("Label", axis=1).replace({'yes': 1, 'no': 0})
y = df["Label"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Save all required files
joblib.dump(model, "disease_model.pkl")         # Model
joblib.dump(list(X.columns), "symptoms.pkl")    # Symptoms list
joblib.dump(accuracy, "accuracy.pkl")           # Accuracy score
