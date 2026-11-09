# INSTAMART DATA ANALYTICS AND CUSTOMER INTELLIGENCE SYSTEM

## 1. Introduction

The Instamart Data Analytics and Customer Intelligence System is a Python-based project developed to analyze customer behaviour and improve customer service operations.

The project uses Data Analytics, Machine Learning, Natural Language Processing (NLP), and database technologies to process customer information, understand purchasing patterns, predict possible customer churn, and classify customer complaints.

## 2. Objectives

- To generate and analyze customer datasets.
- To clean and preprocess raw data.
- To store structured data in an SQLite database.
- To identify customer segments using RFM analysis.
- To predict potential customer churn.
- To classify customer complaints automatically.
- To retrieve relevant company policies using a basic RAG approach.
- To generate reports and visualizations.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning and NLP |
| SQLite | Database management |
| SQLAlchemy | Database connection |

## 4. System Workflow

The project follows these stages:

1. Dataset Generation
2. Data Cleaning
3. Database Creation
4. RFM Customer Segmentation
5. Customer Churn Prediction
6. NLP Complaint Classification
7. Policy Retrieval using RAG
8. Report Generation

## 5. Module Description

### 5.1 Dataset Generation

Sample datasets are generated for customers, orders, complaints, and company policies.

### 5.2 Data Cleaning

The raw datasets are processed by handling missing values, removing duplicate records, standardizing city and gender fields, and managing outliers.

### 5.3 Database Management

The cleaned datasets are stored in an SQLite database named Instamart_DB.

### 5.4 RFM Customer Segmentation

RFM analysis uses three important measures:

- Recency: How recently a customer purchased.
- Frequency: How often a customer purchased.
- Monetary: How much a customer spent.

K-Means clustering is used to group customers based on these measures.

### 5.5 Customer Churn Prediction

A Random Forest model is used to demonstrate how customer information can help identify customers who may stop purchasing.

### 5.6 NLP Complaint Classification

Text classification is used to categorize customer complaints into groups such as delivery, refund, payment, product, and cancellation.

### 5.7 RAG Policy Retrieval

A basic Retrieval-Augmented Generation approach uses TF-IDF and cosine similarity to find relevant company policies and generate template-based responses.

## 6. Expected Outputs

The project generates:

- Raw and cleaned CSV datasets.
- SQLite database.
- Customer segmentation report.
- Customer cluster visualization.
- PCA variance visualization.
- Churn prediction report.
- Complaint classification report.
- Policy retrieval results.
- Saved machine learning models.

## 7. How to Run the Project

### Step 1: Create a virtual environment

```bash
python -m venv venv
```

### Step 2: Activate the environment

```powershell
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r Requirements.txt
```

### Step 4: Run the complete project

```bash
python run_project.py
```

## 8. Conclusion

The Instamart Data Analytics and Customer Intelligence System demonstrates how data analytics, machine learning, NLP, and database technologies can be integrated into one workflow.

It provides a learning-based demonstration of customer segmentation, churn prediction, complaint classification, and policy retrieval. The project uses generated sample data, so its predictions and reports should not be treated as verified results from a real Instamart business.