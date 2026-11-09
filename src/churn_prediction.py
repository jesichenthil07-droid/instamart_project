
import pandas as pd
import sqlite3
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = BASE_DIR / "database" / "Instamart_DB.db"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def build_churn_model():

    # 1. Load order data
    conn = sqlite3.connect(DB_PATH)

    orders = pd.read_sql(
        "SELECT * FROM orders", conn
    )

    conn.close()

    orders["order_date"] = pd.to_datetime(
        orders["order_date"]
    )

    # 2. Calculate customer features
    reference_date = orders["order_date"].max() + pd.Timedelta(days=1)

    customer_data = orders.groupby("customer_id").agg(
        Recency=("order_date", lambda x:
                 (reference_date - x.max()).days),
        Frequency=("order_id", "count"),
        Monetary=("order_amount", "sum"),
        Average_Order_Value=("order_amount", "mean")
    ).reset_index()

    # 3. Create a simulated churn label
    # Customers with no order in the last 14 days
    # are labelled as potential churn customers.
    customer_data["Churn"] = (
        customer_data["Recency"] > 14
    ).astype(int)

    print("\nCustomer Churn Distribution:")
    print(customer_data["Churn"].value_counts())

    # 4. Select prediction features
    # Recency is excluded because it directly creates
    # our simulated churn label.
    features = [
        "Frequency",
        "Monetary",
        "Average_Order_Value"
    ]

    X = customer_data[features]
    y = customer_data["Churn"]

    if y.nunique() < 2:
        raise ValueError(
            "Only one churn class found. "
            "Adjust the simulated churn threshold."
        )

    # 5. Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 6. Build Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    # 7. Train model
    model.fit(X_train, y_train)

    # 8. Predict churn
    predictions = model.predict(X_test)

    # 9. Evaluate model
    accuracy = accuracy_score(y_test, predictions)

    print("\nChurn Prediction Results")
    print("------------------------")

    print("Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # 10. Save model
    joblib.dump(
        model,
        MODEL_DIR / "churn_model.pkl"
    )

    # 11. Save customer predictions
    customer_data["Predicted_Churn"] = model.predict(X)

    customer_data.to_csv(
        REPORT_DIR / "churn_predictions.csv",
        index=False
    )

    print("\nChurn model saved successfully!")
    print("Churn predictions saved successfully!")


if __name__ == "__main__":
    build_churn_model()
