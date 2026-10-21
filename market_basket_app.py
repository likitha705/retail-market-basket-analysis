import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(
    page_title="Retail Market Basket Analysis",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Retail Market Basket Analysis")
st.caption(
    "Frequent Pattern Mining • Association Rules • "
    "Basic Data Deduplication and Product Name Normalization"
)

st.write(
    "Upload a transaction CSV with two columns: "
    "`TransactionID` and `Item`."
)

with st.expander("View sample CSV format"):
    sample = pd.DataFrame(
        {
            "TransactionID": ["T001", "T001", "T001", "T002", "T002", "T003", "T003"],
            "Item": ["Milk", "Bread", "Butter", "Milk", "Bread", "Bread", "Butter"],
        }
    )
    st.dataframe(sample, use_container_width=True)
    st.download_button(
        "Download sample CSV",
        data=sample.to_csv(index=False).encode("utf-8"),
        file_name="sample_transactions.csv",
        mime="text/csv",
    )

uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

if uploaded_file is not None:
    try:
        original_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        st.stop()

    required_columns = {"TransactionID", "Item"}
    if not required_columns.issubset(original_df.columns):
        st.error(
            "Missing required columns. Your CSV must include "
            "`TransactionID` and `Item` (case-sensitive)."
        )
        st.write("Columns found:", list(original_df.columns))
        st.stop()

    st.subheader("1. Original Dataset")
    st.write(f"Rows: {len(original_df):,}")
    st.dataframe(original_df.head(50), use_container_width=True)

    df = original_df.copy()
    rows_before = len(df)
    df = df.dropna(subset=["TransactionID", "Item"]).copy()
    df["TransactionID"] = df["TransactionID"].astype(str).str.strip()
    df["Item"] = df["Item"].astype(str).str.strip().str.lower()
    df = df[
        (df["TransactionID"] != "")
        & (df["Item"] != "")
        & (df["TransactionID"].str.lower() != "nan")
        & (df["Item"].str.lower() != "nan")
    ].copy()

    rows_after_missing = len(df)
    df = df.drop_duplicates(subset=["TransactionID", "Item"]).copy()
    duplicates_removed = rows_after_missing - len(df)
    missing_or_blank_removed = rows_before - rows_after_missing

    st.subheader("2. Data Cleaning and Deduplication")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{df['TransactionID'].nunique():,}")
    c2.metric("Unique Products", f"{df['Item'].nunique():,}")
    c3.metric("Duplicate rows removed", f"{duplicates_removed:,}")
    c4.metric("Blank/missing rows removed", f"{missing_or_blank_removed:,}")

    st.info(
        "Product-name normalization here trims whitespace and converts names "
        "to lowercase. It does not automatically merge spelling variations "
        "such as 'milk' and 'milkk'; advanced graph-based entity resolution "
        "would need an additional matching module."
    )

    with st.expander("View cleaned dataset"):
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download cleaned CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_transactions.csv",
            mime="text/csv",
        )

    if df.empty:
        st.warning("No valid transaction rows remain after cleaning.")
        st.stop()

    basket = pd.crosstab(df["TransactionID"], df["Item"]).gt(0)

    st.subheader("3. Configure Pattern Mining")
    left, right = st.columns(2)
    with left:
        min_support = st.slider(
            "Minimum support",
            min_value=0.01,
            max_value=1.00,
            value=0.05,
            step=0.01,
            help="Fraction of transactions containing an itemset.",
        )
    with right:
        min_confidence = st.slider(
            "Minimum confidence",
            min_value=0.01,
            max_value=1.00,
            value=0.30,
            step=0.01,
            help="Minimum confidence for association rules.",
        )

    if st.button("Run Market Basket Analysis", type="primary"):
        try:
            with st.spinner("Finding frequent itemsets and association rules..."):
                frequent_itemsets = apriori(
                    basket,
                    min_support=min_support,
                    use_colnames=True,
                )

                st.subheader("4. Frequent Itemsets")
                if frequent_itemsets.empty:
                    st.warning(
                        "No frequent itemsets found. Try lowering minimum support."
                    )
                else:
                    itemsets_display = frequent_itemsets.copy()
                    itemsets_display["itemsets"] = itemsets_display[
                        "itemsets"
                    ].apply(lambda items: ", ".join(sorted(items)))
                    itemsets_display = itemsets_display.sort_values(
                        "support", ascending=False
                    )
                    st.dataframe(itemsets_display, use_container_width=True)
                    st.download_button(
                        "Download frequent itemsets CSV",
                        data=itemsets_display.to_csv(index=False).encode("utf-8"),
                        file_name="frequent_itemsets.csv",
                        mime="text/csv",
                    )

                st.subheader("5. Association Rules")
                if frequent_itemsets.empty or frequent_itemsets["itemsets"].map(len).max() < 2:
                    st.info(
                        "There are no frequent itemsets containing two or more "
                        "products, so association rules cannot be generated. "
                        "Try lowering minimum support."
                    )
                else:
                    rules = association_rules(
                        frequent_itemsets,
                        metric="confidence",
                        min_threshold=min_confidence,
                    )

                    if rules.empty:
                        st.info(
                            "No rules met this confidence threshold. "
                            "Try lowering minimum confidence."
                        )
                    else:
                        rules_display = rules.copy()
                        rules_display["antecedents"] = rules_display[
                            "antecedents"
                        ].apply(lambda items: ", ".join(sorted(items)))
                        rules_display["consequents"] = rules_display[
                            "consequents"
                        ].apply(lambda items: ", ".join(sorted(items)))
                        columns = [
                            "antecedents",
                            "consequents",
                            "support",
                            "confidence",
                            "lift",
                        ]
                        rules_display = rules_display[columns].sort_values(
                            "lift", ascending=False
                        )
                        st.dataframe(rules_display, use_container_width=True)
                        st.download_button(
                            "Download association rules CSV",
                            data=rules_display.to_csv(index=False).encode("utf-8"),
                            file_name="association_rules.csv",
                            mime="text/csv",
                        )

                        st.subheader("6. How to interpret the results")
                        st.markdown(
                            "- **Support:** fraction of transactions containing both sides of a rule.\n"
                            "- **Confidence:** fraction of transactions with the left-side item(s) that also contain the right-side item(s).\n"
                            "- **Lift:** how much more often the items occur together than expected if they were independent. Lift above 1 indicates a positive association."
                        )
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.caption(
                "Check that the CSV has valid transactions and try lowering "
                "the support threshold if the dataset is small."
            )
else:
    st.info("Upload a CSV to start the analysis.")
