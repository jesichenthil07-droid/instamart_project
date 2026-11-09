
import pandas as pd
from pathlib import Path

# Project folder location
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
CLEAN_DIR = BASE_DIR / "data" / "cleaned"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)


# Function to remove duplicate records
def remove_duplicates(df, column):
    before = len(df)
    df = df.drop_duplicates(subset=column)
    removed = before - len(df)
    print("Duplicates removed:", removed)
    return df


# 1. Clean customers dataset
def clean_customers():

    df = pd.read_csv(RAW_DIR / "customers.csv")

    df = remove_duplicates(df, "customer_id")

    df["customer_name"] = df["customer_name"].fillna("Unknown")

    df["city"] = df["city"].fillna("Unknown")
    df["city"] = df["city"].astype(str).str.strip().str.title()

    df["gender"] = df["gender"].fillna("Unknown")
    df["gender"] = df["gender"].astype(str).str.strip().str.title()

    df.to_csv(CLEAN_DIR / "customers_cleaned.csv", index=False)

    print("Customers cleaned successfully!")


# 2. Clean orders dataset
def clean_orders():

    df = pd.read_csv(RAW_DIR / "orders.csv")

    df = remove_duplicates(df, "order_id")

    df["order_amount"] = pd.to_numeric(
        df["order_amount"], errors="coerce"
    )

    df["order_amount"] = df["order_amount"].fillna(
        df["order_amount"].median()
    )

    # Remove invalid negative order amounts
    df = df[df["order_amount"] >= 0].copy()

    # Handle outliers using IQR capping
    q1 = df["order_amount"].quantile(0.25)
    q3 = df["order_amount"].quantile(0.75)

    iqr = q3 - q1

    lower = max(0, q1 - 1.5 * iqr)
    upper = q3 + 1.5 * iqr

    df["order_amount"] = df["order_amount"].clip(
        lower=lower, upper=upper
    )

    df["order_date"] = pd.to_datetime(
        df["order_date"], errors="coerce"
    )

    df = df.dropna(subset=["order_date", "customer_id"])

    df.to_csv(CLEAN_DIR / "orders_cleaned.csv", index=False)

    print("Orders cleaned successfully!")


# 3. Clean tickets dataset
def clean_tickets():

    df = pd.read_csv(RAW_DIR / "tickets.csv")

    df = remove_duplicates(df, "ticket_id")

    df["complaint"] = df["complaint"].fillna(
        "Unknown complaint"
    )

    df["complaint"] = df["complaint"].astype(str).str.strip()

    df = df.dropna(subset=["customer_id"])

    df.to_csv(CLEAN_DIR / "tickets_cleaned.csv", index=False)

    print("Tickets cleaned successfully!")


# 4. Clean policies dataset
def clean_policies():

    df = pd.read_csv(RAW_DIR / "policies.csv")

    df = remove_duplicates(df, "policy_id")

    df["policy_name"] = df["policy_name"].fillna("Unknown")
    df["policy_text"] = df["policy_text"].fillna(
        "Policy information unavailable"
    )

    df.to_csv(CLEAN_DIR / "policies_cleaned.csv", index=False)

    print("Policies cleaned successfully!")


# Run all cleaning functions
if __name__ == "__main__":

    print("\nSTARTING DATA CLEANING\n")

    clean_customers()
    clean_orders()
    clean_tickets()
    clean_policies()

    print("\nALL DATASETS CLEANED SUCCESSFULLY!")
