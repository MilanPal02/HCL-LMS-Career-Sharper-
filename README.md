# Banking Transaction Pattern Analysis

**Project 10 | Domain:** Banking & FinTech  
**Program:** HCL Tech – LMS Career Sharper  

---

## Student Details
* **Name:** Milan Pal
* **Admission Number:** 2026B01010203
* **Course:** B.Tech (CSE-9) – 1st Year
* **Repository:** [HCL-LMS-Career-Sharper-]([https://github.com](https://github.com/MilanPal02/HCL-LMS-Career-Sharper-.git))

---

## Project Overview
This project inspects, cleans, transforms, and analyzes synthetic banking transaction records using **Python**, **NumPy**, and **Pandas**. It identifies transaction trends across payment channels, cities, and account types while handling missing data, detecting statistical outliers, and categorizing transactions into value bands.

---

## Repository Structure
```text
├── MyProject.py                        # Main Python analysis and data cleaning script
├── 10_bank_transactions.csv            # Raw synthetic banking dataset
├── 10_bank_transactions_cleaned.csv    # Processed dataset exported after cleaning
├── SRS                                 # Software Requirements Specification document
└── README.md                           # Project documentation
```

---

## Key Features & Analysis Steps
1. **Data Inspection & Formatting:** Loads the dataset, checks structural dimensions and data types, and converts `Transaction_Date` into standard `datetime` format.
2. **Data Cleaning & Imputation:** Identifies duplicate `Transaction_ID` entries and fills missing `Amount_INR` values using the grouped median (by `Transaction_Type` and `Channel`) to prevent outlier distortion.
3. **Credit vs. Debit Summary:** Aggregates transaction counts and total rupee value across Credit and Debit transactions.
4. **Statistical Profiling (NumPy):** Computes mean, median, and standard deviation on transaction amounts.
5. **Channel, City & Account Breakdown:** Evaluates transaction volumes and values across payment channels, cities, and account types.
6. **Flagged Transaction Review:** Calculates overall and channel-wise percentages of transactions flagged for review.
7. **Outlier Detection (IQR Method):** Uses 25th ($Q_1$) and 75th ($Q_3$) percentiles to compute the Interquartile Range and detect extreme transaction values.
8. **Amount Banding & Threshold Filtering:** Categorizes transactions into four value tiers (`Micro`, `Standard`, `High`, `Very High`) using `pd.cut()` and filters high-value transactions (`> INR 2,500`) via NumPy boolean indexing.

---

## Technologies Used
* **Python 3.x**
* **Pandas** – Data loading, cleaning, grouping, and transformation
* **NumPy** – Numerical calculations, percentiles, and boolean array filtering

---

## How to Run
1. **Clone the repository:**
   ```bash
   git clone https://github.com.git
   cd HCL-LMS-Career-Sharper-
   ```

2. **Install required libraries:**
   ```bash
   pip install numpy pandas
   ```

3. **Run the script:**
   ```bash
   python MyProject.py
   ```
   *This will display the complete statistical summary in the terminal and generate `10_bank_transactions_cleaned.csv`.*
