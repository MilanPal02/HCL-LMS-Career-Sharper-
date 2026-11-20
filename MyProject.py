#Name: Milan Pal
#Addmission Number: 2026B01010203
#B.Tech (CSE-9) 1st Year
#Github Repository Link: https://github.com/MilanPal02/HCL-LMS-Career-Sharper-.git
#Project 10: Banking Transaction Pattern Analysis
#Industry: Banking / FinTech
#Description: Inspects, cleans, transforms, and analyzes synthetic banking transaction records 
#using Python, NumPy, and Pandas.


import numpy as np
import pandas as pd

# ==========================================
# 1. Load and Inspect Transaction Records
# ==========================================
def load_and_inspect_data(filepath):
    """Loads the CSV dataset and prints basic structural summaries."""
    df = pd.read_csv(filepath)
    print("--- 1. DATASET OVERVIEW ---")
    print(f"Shape (Rows, Columns): {df.shape}")
    print("\nColumn Data Types & Non-Null Counts:")
    print(df.info())
    print("\nFirst 5 Records:")
    print(df.head())
    return df

# Load dataset (Update path if placed in a /data/ subfolder)
df = load_and_inspect_data("10_bank_transactions.csv")

# Convert Transaction_Date to datetime format for clean date handling
df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"])

# ==========================================
# 2. Missing Values & Duplicate Verification
# ==========================================
print("\n--- 2. DATA QUALITY CHECKS ---")
missing_counts = df.isnull().sum()
print("Missing Values per Column:\n", missing_counts)

duplicate_ids = df["Transaction_ID"].duplicated().sum()
print(f"\nDuplicate Transaction_ID Count: {duplicate_ids}")

# Handle missing Amount_INR values:
# Strategy: Impute missing transaction amounts using the median amount grouped by 
# Transaction_Type and Channel. Median is chosen over mean to prevent extreme outliers 
# from skewing imputed values.
df["Amount_INR"] = df["Amount_INR"].fillna(
    df.groupby(["Transaction_Type", "Channel"])["Amount_INR"].transform("median")
)
print(f"Missing Amount_INR after median imputation: {df['Amount_INR'].isnull().sum()}")

# ==========================================
# 3. Total Credits and Total Debits
# ==========================================
print("\n--- 3. CREDIT VS. DEBIT TOTALS ---")
type_totals = df.groupby("Transaction_Type")["Amount_INR"].agg(
    Transaction_Count="count",
    Total_Amount_INR="sum"
)
print(type_totals)

# ==========================================
# 4. Statistical Summary using NumPy
# ==========================================
print("\n--- 4. NUMPY STATISTICAL SUMMARY (AMOUNT_INR) ---")
amounts_array = df["Amount_INR"].to_numpy()

mean_amt = np.mean(amounts_array)
median_amt = np.median(amounts_array)
std_amt = np.std(amounts_array)

print(f"Mean Transaction Amount:   INR {mean_amt:,.2f}")
print(f"Median Transaction Amount: INR {median_amt:,.2f}")
print(f"Standard Deviation:        INR {std_amt:,.2f}")

# ==========================================
# 5. Transaction Volume by Channel
# ==========================================
print("\n--- 5. TRANSACTION VOLUME BY CHANNEL ---")
channel_summary = df.groupby("Channel").agg(
    Transaction_Volume=("Transaction_ID", "count"),
    Total_Value_INR=("Amount_INR", "sum"),
    Average_Value_INR=("Amount_INR", "mean")
).sort_values(by="Transaction_Volume", ascending=False)
print(channel_summary)

# ==========================================
# 6. Total Transaction Value by City
# ==========================================
print("\n--- 6. TOTAL TRANSACTION VALUE BY CITY ---")
city_summary = df.groupby("City").agg(
    Transaction_Count=("Transaction_ID", "count"),
    Total_Value_INR=("Amount_INR", "sum")
).sort_values(by="Total_Value_INR", ascending=False)
print(city_summary)

# ==========================================
# 7. Flagged Transaction Analysis
# ==========================================
print("\n--- 7. FLAGGED TRANSACTION SUMMARY ---")
# Note: 'Flagged' is a supplied synthetic label for review, not proof of fraud.
total_tx = len(df)
flagged_count = (df["Flagged"] == "Yes").sum()
flagged_pct = (flagged_count / total_tx) * 100

print(f"Total Flagged Transactions: {flagged_count} out of {total_tx} ({flagged_pct:.2f}%)")

print("\nFlagged Percentage Across Channels:")
channel_flagged = pd.crosstab(df["Channel"], df["Flagged"], normalize="index") * 100
print(channel_flagged.round(2))

# ==========================================
# 8. Top 10 Largest Transactions
# ==========================================
print("\n--- 8. TOP 10 LARGEST TRANSACTIONS ---")
top_10_tx = df.sort_values(by="Amount_INR", ascending=False).head(10)
print(top_10_tx[["Transaction_ID", "Account_Type", "Transaction_Date", "Transaction_Type", "Amount_INR", "Channel", "City", "Flagged"]])

# ==========================================
# 9. Credit & Debit Totals by Account Type
# ==========================================
print("\n--- 9. CREDIT & DEBIT TOTALS BY ACCOUNT TYPE ---")
account_type_summary = df.groupby(["Account_Type", "Transaction_Type"])["Amount_INR"].agg(
    Count="count",
    Total_INR="sum",
    Mean_INR="mean"
)
print(account_type_summary)

# ==========================================
# 10. Outlier Detection using IQR (NumPy)
# ==========================================
print("\n--- 10. OUTLIER DETECTION (IQR METHOD) ---")
q1 = np.percentile(amounts_array, 25)
q3 = np.percentile(amounts_array, 75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df[(df["Amount_INR"] < lower_bound) | (df["Amount_INR"] > upper_bound)]
print(f"Q1: {q1:.2f} | Q3: {q3:.2f} | IQR: {iqr:.2f}")
print(f"Upper Outlier Threshold: INR {upper_bound:.2f}")
print(f"Total Outlier Transactions Detected: {len(outliers)}")

# ==========================================
# 11. Transaction Amount Bands
# ==========================================
print("\n--- 11. TRANSACTION AMOUNT BANDS ---")
bins = [0, 250, 1000, 3000, np.inf]
labels = ["Micro (<=250)", "Standard (251-1000)", "High (1001-3000)", "Very High (>3000)"]

df["Amount_Band"] = pd.cut(df["Amount_INR"], bins=bins, labels=labels)
band_distribution = df["Amount_Band"].value_counts().sort_index()
print(band_distribution)

# ==========================================
# 12. NumPy Boolean Indexing Above Threshold
# ==========================================
print("\n--- 12. NUMPY BOOLEAN INDEXING (THRESHOLD > INR 2,500) ---")
threshold = 2500.0
high_value_mask = amounts_array > threshold
high_value_amounts = amounts_array[high_value_mask]

print(f"Transactions strictly above INR {threshold:,.2f}: {len(high_value_amounts)}")
print(f"Combined Value of High-Value Transactions: INR {np.sum(high_value_amounts):,.2f}")

# Export cleaned dataset for repository deliverables
df.to_csv("10_bank_transactions_cleaned.csv", index=False)
print("\nCleaned dataset saved to '10_bank_transactions_cleaned.csv'.")
