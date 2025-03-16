import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
DATASET_PATH = os.path.join(BASE_DIR, "construction_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "ml_model.pkl")


def train_model():
    # Load dataset
    data = pd.read_csv(DATASET_PATH)

    # Encode categorical variables
    label_encoders = {}
    for col in ["CementQuality", "BrickQuality", "SandQuality", "IronQuality", "EnvironmentalCondition", "Seller"]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Features & Target
    X = data[["CementQuality",  "BrickQuality",
              "SandQuality",  "IronQuality",  "EnvironmentalCondition"]]
    y = data["Price"]

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save Model & Encoders
    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoders, os.path.join(BASE_DIR, "label_encoders.pkl"))

    print("Model trained and saved.")


def predict_price(cement_quality, brick_quality, sand_quality, iron_quality, env_condition):
    # Load model & encoders
    model = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(os.path.join(BASE_DIR, "label_encoders.pkl"))

    # Encode categorical inputs
    cement_quality = label_encoders["CementQuality"].transform([cement_quality])[0]
    brick_quality = label_encoders["BrickQuality"].transform([brick_quality])[0]
    sand_quality = label_encoders["SandQuality"].transform([sand_quality])[0]
    iron_quality = label_encoders["IronQuality"].transform([iron_quality])[0]
    env_condition = label_encoders["EnvironmentalCondition"].transform([env_condition])[0]

    # Predict
    predicted_price = model.predict([[cement_quality,  brick_quality,
                                      sand_quality, iron_quality,  env_condition]])

    return round(predicted_price[0], 2)


if __name__ == "__main__":
    train_model()
