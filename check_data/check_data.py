import pandas as pd

customers = pd.read_csv("data/raw/customers.csv")
orders = pd.read_csv("data/raw/orders.csv")
tickets = pd.read_csv("data/raw/tickets.csv")
policies = pd.read_csv("data/raw/policies.csv")

print("CUSTOMERS")
print(customers.head())

print("\nORDERS")
print(orders.head())

print("\nTICKETS")
print(tickets.head())

print("\nPOLICIES")
print(policies.head())