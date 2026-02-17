import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(csv_path):
    df = pd.read_csv(csv_path)

    X = df.drop("Stress_level", axis=1)
    y = df["Stress_level"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "stress_model.pkl")
    print("Model saved as stress_model.pkl")

    return model, X_test, y_test
