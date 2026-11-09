
import pandas as pd
import sqlite3
import json
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = BASE_DIR / "database" / "Instamart_DB.db"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# 1. Training examples for NLP classification
training_data = [
    ("My order is delayed", "Delivery"),
    ("Where is my order", "Delivery"),
    ("Delivery is taking too long", "Delivery"),
    ("My package has not arrived", "Delivery"),

    ("I want a refund", "Refund"),
    ("Please return my money", "Refund"),
    ("Refund is not received", "Refund"),
    ("I need my money back", "Refund"),

    ("Payment failed", "Payment"),
    ("Money deducted but order not placed", "Payment"),
    ("Payment transaction unsuccessful", "Payment"),
    ("Unable to pay online", "Payment"),

    ("I received a damaged product", "Product"),
    ("Wrong item delivered", "Product"),
    ("Product is broken", "Product"),
    ("I received a defective item", "Product"),

    ("I want to cancel my order", "Cancellation"),
    ("Please cancel my order", "Cancellation"),
    ("Order cancellation request", "Cancellation"),
    ("I do not need this order", "Cancellation")
]


# 2. Train NLP model
def train_ticket_model():

    texts = [item[0] for item in training_data]
    labels = [item[1] for item in training_data]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2)
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    model.fit(texts, labels)

    joblib.dump(
        model,
        MODEL_DIR / "ticket_model.pkl"
    )

    return model


# 3. Department and priority routing
routing_rules = {
    "Delivery": {
        "priority": "High",
        "department": "Delivery Team"
    },
    "Refund": {
        "priority": "Medium",
        "department": "Refund Team"
    },
    "Payment": {
        "priority": "High",
        "department": "Payment Team"
    },
    "Product": {
        "priority": "High",
        "department": "Replacement Team"
    },
    "Cancellation": {
        "priority": "Medium",
        "department": "Order Management Team"
    }
}


# 4. Classify and route tickets
def process_ticket(ticket, model):

    complaint = str(ticket["complaint"])

    category = model.predict([complaint])[0]

    routing = routing_rules[category]

    ticket_result = {
        "ticket_id": int(ticket["ticket_id"]),
        "customer_id": int(ticket["customer_id"]),
        "complaint": complaint,
        "category": category,
        "priority": routing["priority"],
        "department": routing["department"],
        "status": "Assigned"
    }

    return ticket_result


# 5. Process all tickets
def run_triage():

    conn = sqlite3.connect(DB_PATH)

    tickets = pd.read_sql(
        "SELECT * FROM tickets", conn
    )

    conn.close()

    model = train_ticket_model()

    results = []

    for _, ticket in tickets.iterrows():

        result = process_ticket(ticket, model)

        results.append(result)

    # Save ticket results as CSV
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        REPORT_DIR / "triage_results.csv",
        index=False
    )

    # Save ticket results as JSON
    with open(
        REPORT_DIR / "triage_results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print("\nTICKET TRIAGE COMPLETED!")

    print("\nSample Ticket Result:")

    print(json.dumps(results[0], indent=4))

    print("\nTotal Tickets Processed:", len(results))

    print("Ticket model saved successfully!")
    print("Triage reports saved successfully!")


if __name__ == "__main__":
    run_triage()
