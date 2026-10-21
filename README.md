# Retail Market Basket Analysis

A Streamlit application for a Data Mining & Warehousing project on frequent pattern mining and association rule mining for supermarket transactions.

## Features

- Upload a supermarket transaction CSV.
- Preview the original dataset.
- Clean blank/missing transaction and item values.
- Normalize product names by trimming whitespace and converting to lowercase.
- Remove duplicate transaction-product records.
- Mine frequent itemsets using the Apriori algorithm.
- Generate association rules with support, confidence, and lift.
- Download cleaned data, frequent itemsets, and association rules as CSV files.

> **Project scope note:** The current version performs basic product-name normalization and exact duplicate removal. It does not yet implement graph-based entity resolution or fuzzy matching of spelling variants; those are extensions for a fuller implementation.

## CSV format

The uploaded CSV must have these exact column names:

```csv
TransactionID,Item
T001,Milk
T001,Bread
T001,Butter
T002,Milk
T002,Bread
T003,Bread
T003,Butter
```

Each row represents one product in one transaction.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. (Recommended) Create and activate a virtual environment:

   **Windows**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **macOS/Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the app:

   ```bash
   streamlit run market_basket_app.py
   ```

6. Open the local URL printed in the terminal (usually `http://localhost:8501`).

## Deploy on Streamlit Community Cloud

1. Upload `market_basket_app.py`, `requirements.txt`, and `README.md` to a GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Create an app and select your repository and `main` branch.
4. Set the main file path to `market_basket_app.py`.
5. Click **Deploy**.

## Metrics

- **Support:** proportion of all transactions containing an itemset.
- **Confidence:** proportion of transactions containing the antecedent that also contain the consequent.
- **Lift:** ratio of observed co-occurrence to the co-occurrence expected if the itemsets were independent.

## Limitations

- Apriori performance may decrease on very large datasets.
- The app expects one product per row and the column names `TransactionID` and `Item`.
- Product normalization is intentionally simple; semantically equivalent names such as `coke`, `coca-cola`, or misspellings require a separate entity-resolution method.
- Do not upload sensitive or personally identifiable customer information to a public repository or demo app.

## Technology stack

Python, Streamlit, pandas, and mlxtend.
