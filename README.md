# Stock Transactions Capital Gains Analysis

## Overview
Python script for analyzing stock transactions and computing capital gains using UK tax reporting rules.

## Features
- Currency conversion using live Yahoo Finance rates
- Applies UK tax calculation rules:
  - Same-day matching
  - Bed-and-breakfast (30-day) rule
  - S104 pool tracking
- Generates comprehensive Excel tax reporting

## Prerequisites
- Python 3.8+
- Required libraries:
  - pandas
  - numpy
  - yfinance
  - openpyxl

## Setup
1. Install dependencies:
```bash
pip install pandas numpy yfinance openpyxl
```

2. Prepare input CSV:
   - Ensure `stock_transactions.csv` follows required transaction posting format
   - Include columns: date, account, commodity, debit, credit, amount, txnidx

## Usage
```bash
python main.py
```

## Output
Generates `stock_transactions_report.xlsx` with:
- All Transactions sheet
- Individual sheets per stock symbol
- Detailed capital gains calculations

## Tax Reporting Period
Configured for UK tax year: April 6, 2023 - April 5, 2024.
Can change this in the program.

## Disclaimer
Use for informational purposes. Consult a tax professional for official guidance.
