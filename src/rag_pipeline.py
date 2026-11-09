
import pandas as pd
import json
import sqlite3

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = BASE_DIR / "database" / "Instamart_DB.db"
REPORT_DIR = BASE_DIR / "reports"


# Step 1: Load policy knowledge base
def load_policies():

    conn = sqlite3.connect(DB_PATH)

    policies = pd.read_sql(
        "SELECT * FROM policies", conn
    )

    conn.close()

    return policies


# Step 2: Build policy retrieval system
def retrieve_policy(complaint, policies):

    documents = policies["policy_text"].fillna("").tolist()

    vectorizer = TfidfVectorizer()

    policy_vectors = vectorizer.fit_transform(documents)

    complaint_vector = vectorizer.transform([complaint])

    similarities = cosine_similarity(
        complaint_vector,
        policy_vectors
    )[0]

    best_index = similarities.argmax()

    best_score = similarities[best_index]

    if best_score == 0:
        return {
            "policy_name": "No matching policy",
            "policy_text": "No relevant policy was found.",
            "similarity_score": 0
        }

    return {
        "policy_name": policies.iloc[best_index]["policy_name"],
        "policy_text": policies.iloc[best_index]["policy_text"],
        "similarity_score": round(float(best_score), 3)
    }


# Step 3: Generate response using mock LLM
def generate_response(ticket, policy):

    from mock_llm import generate_mock_response

    response = generate_mock_response(
        ticket,
        policy
    )

    return response


# Step 4: Connect RAG with ticket triage
def run_rag():

    policies = load_policies()

    triage_path = REPORT_DIR / "triage_results.json"

    with open(triage_path, "r", encoding="utf-8") as file:
        tickets = json.load(file)

    rag_results = []

    for ticket in tickets:

        policy = retrieve_policy(
            ticket["complaint"],
            policies
        )

        response = generate_response(
            ticket,
            policy
        )

        result = {
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "complaint": ticket["complaint"],
            "category": ticket["category"],
            "priority": ticket["priority"],
            "department": ticket["department"],
            "retrieved_policy": policy["policy_name"],
            "policy_information": policy["policy_text"],
            "similarity_score": policy["similarity_score"],
            "response": response
        }

        rag_results.append(result)

    # Save RAG results
    output_path = REPORT_DIR / "rag_results.json"

    with open(output_path, "w", encoding="utf-8") as file:

        json.dump(
            rag_results,
            file,
            indent=4
        )

    # Save CSV report
    pd.DataFrame(rag_results).to_csv(
        REPORT_DIR / "rag_results.csv",
        index=False
    )

    print("\nRAG PIPELINE COMPLETED!")

    print("\nSample RAG Result:")

    print(json.dumps(rag_results[0], indent=4))

    print("\nTotal tickets processed:", len(rag_results))


if __name__ == "__main__":
    run_rag()
