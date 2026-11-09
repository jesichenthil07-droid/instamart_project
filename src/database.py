
import pandas as pd
import sqlite3
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parent.parent

CLEAN_DIR = BASE_DIR / "data" / "cleaned"
DB_DIR = BASE_DIR / "database"

DB_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DB_DIR / "Instamart_DB.db"


def load_to_database():

    conn = sqlite3.connect(DB_PATH)

    datasets = {
        "customers": "customers_cleaned.csv",
        "orders": "orders_cleaned.csv",
        "tickets": "tickets_cleaned.csv",
        "policies": "policies_cleaned.csv"
    }

    for table_name, file_name in datasets.items():

        file_path = CLEAN_DIR / file_name

        df = pd.read_csv(file_path)

        df.to_sql(
            table_name,
            conn,
            if_exists="replace",
            index=False
        )

        print(f"{table_name} table created successfully!")

    conn.close()

    print("\nDatabase created successfully!")


if __name__ == "__main__":
    load_to_database()
