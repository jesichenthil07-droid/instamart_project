
import pandas as pd
import numpy as np
import os

os.makedirs("data/raw", exist_ok=True)

np.random.seed(42)

# Customers
customers = pd.DataFrame({
    "customer_id": range(1, 501),
    "customer_name": [f"Customer_{i}" for i in range(1, 501)],
    "city": np.random.choice(
        ["Chennai", "Coimbatore", "Madurai", "Salem"],
        500
    ),
    "gender": np.random.choice(["Male", "Female"], 500)
})

customers.to_csv("data/raw/customers.csv", index=False)

# Orders
orders = pd.DataFrame({
    "order_id": range(1001, 2001),
    "customer_id": np.random.randint(1, 501, 1000),
    "order_amount": np.random.randint(100, 3000, 1000),
    "order_date": pd.date_range(
        "2025-01-01", periods=1000, freq="h"
    )
})

orders.to_csv("data/raw/orders.csv", index=False)

# Tickets
tickets = pd.DataFrame({
    "ticket_id": range(2001, 2301),
    "customer_id": np.random.randint(1, 501, 300),
    "complaint": np.random.choice([
        "Order delayed",
        "Damaged product",
        "Refund request",
        "Payment failed",
        "Wrong item delivered"
    ], 300)
})

tickets.to_csv("data/raw/tickets.csv", index=False)

# Policies
policies = pd.DataFrame({
    "policy_id": [1, 2, 3, 4],
    "policy_name": [
        "Refund", "Cancellation",
        "Delivery", "Replacement"
    ],
    "policy_text": [
        "Refunds take 5-7 business days.",
        "Orders can be cancelled before dispatch.",
        "Delivery depends on customer location.",
        "Damaged products may qualify for replacement."
    ]
})

policies.to_csv("data/raw/policies.csv", index=False)

print("All datasets created successfully!")
