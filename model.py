import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_predict():
    # Load Iris dataset
    data = load_iris()
    X, y = data.data, data.target

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    preds = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, preds)

    return preds, accuracy, len(y_test)