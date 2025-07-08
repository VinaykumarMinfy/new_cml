# train.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

# Write report
with open("metrics.txt", "w") as f:
    f.write("## 📊 Model Metrics Report\n")
    f.write(f"* Accuracy: {accuracy:.4f}\n")
    f.write(f"* F1 Score: {f1:.4f}\n")
    f.write("![cml](https://cml.dev/watermark.png)\n")
