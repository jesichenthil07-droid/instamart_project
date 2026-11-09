
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = BASE_DIR / "database" / "Instamart_DB.db"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def build_rfm_model():

    # 1. Connect to database
    conn = sqlite3.connect(DB_PATH)

    orders = pd.read_sql(
        "SELECT * FROM orders", conn
    )

    conn.close()

    # 2. Convert order date
    orders["order_date"] = pd.to_datetime(
        orders["order_date"]
    )

    # 3. Calculate RFM values
    reference_date = orders["order_date"].max() + pd.Timedelta(days=1)

    rfm = orders.groupby("customer_id").agg(
        Recency=("order_date", lambda x:
                 (reference_date - x.max()).days),
        Frequency=("order_id", "count"),
        Monetary=("order_amount", "sum")
    ).reset_index()

    print("\nRFM VALUES:")
    print(rfm.head())

    # 4. Scale the RFM values
    features = ["Recency", "Frequency", "Monetary"]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(rfm[features])

    # 5. Apply KMeans clustering
    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    rfm["Cluster"] = kmeans.fit_predict(scaled_data)

    # 6. Calculate silhouette score
    score = silhouette_score(
        scaled_data, rfm["Cluster"]
    )

    print("\nSilhouette Score:", round(score, 3))

    # 7. Perform PCA
    pca = PCA(n_components=2)

    pca_data = pca.fit_transform(scaled_data)

    rfm["PCA1"] = pca_data[:, 0]
    rfm["PCA2"] = pca_data[:, 1]

    variance = pca.explained_variance_ratio_

    print("\nPCA Variance Analysis:")
    print("Component 1:", round(variance[0] * 100, 2), "%")
    print("Component 2:", round(variance[1] * 100, 2), "%")
    print("Total Variance:", round(sum(variance) * 100, 2), "%")

    # 8. Save customer segments
    rfm.to_csv(
        REPORT_DIR / "customer_segments.csv",
        index=False
    )

    # 9. Save models
    joblib.dump(
        scaler, MODEL_DIR / "rfm_scaler.pkl"
    )

    joblib.dump(
        kmeans, MODEL_DIR / "kmeans_model.pkl"
    )

    joblib.dump(
        pca, MODEL_DIR / "pca_model.pkl"
    )

    # 10. Create cluster visualization
    plt.figure(figsize=(8, 6))

    plt.scatter(
        rfm["PCA1"],
        rfm["PCA2"],
        c=rfm["Cluster"],
        cmap="viridis"
    )

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Instamart Customer Segmentation")

    plt.savefig(
        REPORT_DIR / "customer_clusters.png"
    )

    plt.close()

    # 11. Save PCA variance chart
    plt.figure(figsize=(7, 5))

    plt.bar(
        ["PCA 1", "PCA 2"],
        variance
    )

    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Variance Analysis")

    plt.savefig(
        REPORT_DIR / "pca_variance.png"
    )

    plt.close()

    print("\nCustomer segmentation completed!")
    print("Customer segments saved successfully!")
    print("Models saved successfully!")
    print("Graphs saved successfully!")


if __name__ == "__main__":
    build_rfm_model()
