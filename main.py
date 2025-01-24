from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 2000)

np.set_printoptions(legacy="1.25")

# Load the CSV data
file_path = "stock_transactions.csv"
postings_df = pd.read_csv(file_path)


def fetch_conversion_rate(date, base_currency, target_currency):
    """
    Fetch the conversion rate from base_currency to target_currency on a specific date.
    Retry for the previous trading day if no data is available.
    """
    if base_currency == target_currency:
        return 1  # No conversion needed if currencies match

    currency_pair = f"{base_currency}{target_currency}=X"
    ticker = yf.Ticker(currency_pair)

    while True:
        try:
            # Ensure date is in datetime format
            date = pd.to_datetime(date)
            start_date = date.strftime("%Y-%m-%d")
            end_date = (date + timedelta(days=1)).strftime("%Y-%m-%d")

            # Fetch historical data
            historical_data = ticker.history(start=start_date, end=end_date)

            if not historical_data.empty:
                return historical_data["Close"].iloc[0]  # Return the closing rate
            else:
                print(
                    f"No data available for {currency_pair} on {start_date}. Trying previous day."
                )
                date -= timedelta(days=1)  # Retry with the previous day
        except Exception as e:
            print(
                f"Error fetching conversion rate for {currency_pair} on {start_date}: {e}"
            )
            return None


def transform_to_transaction_representation(postings_df):
    """
    Convert posting representation to transaction representation.
    """
    # Group by txnidx to pair postings
    transactions = []
    grouped = postings_df.groupby("txnidx")

    for txnidx, group in grouped:
        # Ensure there are exactly two rows per transaction
        if len(group) != 2:
            raise ValueError(
                f"Transaction {txnidx} does not have exactly two postings."
            )

        # Extract rows
        posting1, posting2 = group.iloc[0], group.iloc[1]

        # Identify the stock and cash postings
        if (
            "positions" in posting1["account"].lower()
            or "vested-gsu" in posting1["account"].lower()
        ):
            stock_posting, cash_posting = posting1, posting2
        else:
            stock_posting, cash_posting = posting2, posting1

        # Extract transaction details
        date = stock_posting["date"]  # Date from stock posting
        platform = (
            "IG Share Dealing"
            if "ig-share" in stock_posting["account"].lower()
            else "Morgan Stanley"
        )
        stock_symbol = stock_posting["commodity"]

        # Determine transaction type and quantity
        quantity = abs(
            stock_posting["debit"]
            if pd.notnull(stock_posting["debit"])
            else stock_posting["amount"]
        )
        txn_type = "buy" if stock_posting["debit"] > 0 else "sell"

        # Check if it's a GSU release transaction
        if cash_posting["account"].lower() == "income:salary:gsu":
            txn_currency = cash_posting["commodity"]  # Curfency from cash posting
            txn_amount = abs(
                cash_posting["credit"]
                if pd.notnull(cash_posting["credit"])
                else cash_posting["amount"]
            )
            txn_unit_price = round(txn_amount / quantity, 2) if quantity != 0 else None
        else:
            txn_currency = cash_posting["commodity"]  # Use cash posting's commodity
            txn_amount = abs(
                cash_posting["credit"]
                if pd.notnull(cash_posting["credit"])
                else cash_posting["amount"]
            )
            txn_unit_price = round(txn_amount / quantity, 2) if quantity != 0 else None

        txn_currency = {"$": "USD", "£": "GBP"}[txn_currency]

        # Fetch the conversion rate
        conversion_rate = fetch_conversion_rate(date, txn_currency, "GBP")

        # Append transaction data
        transactions.append(
            {
                "txnidx": txnidx,
                "date": pd.to_datetime(date),
                "trading_platform": platform,
                "stock_symbol": stock_symbol,
                "txn_type": txn_type,
                "txn_currency": txn_currency,
                "quantity": quantity,
                "txn_unit_price": txn_unit_price,
                "txn_amount": txn_amount,
                "conversion_rate": conversion_rate,
                "txn_unit_price_gbp": round(txn_unit_price * conversion_rate, 2),
                "txn_amount_gbp": round(txn_amount * conversion_rate, 2),
                "note": "",
            }
        )

    return pd.DataFrame(transactions)


# Convert "allocations" into "allocations_note"
def convert_allocations_to_note(cell):
    if isinstance(cell, list):  # Ensure the cell is a list
        return "\n".join(
            [
                str(
                    {
                        k: float(v) if isinstance(v, np.float64) else v
                        for k, v in d.items()
                    }
                )
                for d in cell
            ]
        )  # Convert each dictionary to plain text with floats and join with newlines
    return str(cell)  # Convert non-list to string


def compute_capital_gains_with_pool(transactions):
    """
    Compute capital gains for each transaction while maintaining and recording
    the S104 pool state.
    """
    # Sort transactions by date
    transactions = transactions.sort_values(by="date").reset_index(drop=True)

    # Initialize S104 pool
    s104_pool = {"quantity": 0.0, "total_cost": 0.0}

    # Add new columns
    transactions["allocations"] = [[] for _ in range(len(transactions))]
    transactions["s104_pre_txn"] = None
    transactions["s104_post_txn"] = None
    transactions["capital_pnl"] = None
    transactions["unallocated_qty"] = transactions["quantity"]

    # Identify all same-day allocations
    # Same-day rule needs to be considered comprehensively for all transactions
    # before the b-n-b and s104 computation.
    # This means if a buy txn can match with both a previous 30-day sell
    # transaction and a later-in-the-day sell transaction, it should still
    # first match the same-day sell txn because that takes precedence.
    for idx, _ in transactions.iterrows():
        if transactions.at[idx, "txn_type"] == "sell":
            same_day_buys = transactions[
                (transactions["date"] == transactions.at[idx, "date"])
                & (transactions["txn_type"] == "buy")
            ]

            for buy_idx, _ in same_day_buys.iterrows():
                if transactions.at[idx, "unallocated_qty"] <= 0:
                    break

                if transactions.at[buy_idx, "unallocated_qty"] <= 0:
                    continue

                match_qty = round(
                    min(
                        transactions.at[idx, "unallocated_qty"],
                        transactions.at[buy_idx, "unallocated_qty"],
                    ),
                    3,
                )

                if match_qty != 0:  # avoid float computation error
                    transactions.at[idx, "allocations"].append(
                        {
                            "rule": "same-day",
                            "matched_quantity": match_qty,
                            "matched_cost": round(
                                transactions.at[buy_idx, "txn_amount_gbp"]
                                / transactions.at[buy_idx, "quantity"]
                                * match_qty,
                                2,
                            ),
                            "matched_txnid": int(transactions.at[buy_idx, "txnidx"]),
                        }
                    )

                    transactions.at[buy_idx, "allocations"].append(
                        {
                            "rule": "same-day",
                            "matched_quantity": match_qty,
                            "matched_txnid": int(transactions.at[idx, "txnidx"]),
                        }
                    )

                    transactions.at[idx, "unallocated_qty"] -= match_qty
                    transactions.at[buy_idx, "unallocated_qty"] -= match_qty

    # Process each transaction
    for idx, _ in transactions.iterrows():
        if (
            transactions.at[idx, "txn_type"] == "buy"
            or transactions.at[idx, "txn_type"] == "split"
        ):
            if round(transactions.at[idx, "unallocated_qty"], 3) > 0:
                transactions.at[idx, "allocations"].append(
                    {
                        "rule": "s104",
                        "matched_quantity": round(
                            transactions.at[idx, "unallocated_qty"], 3
                        ),
                    }
                )

                transactions.at[idx, "s104_pre_txn"] = (
                    round(s104_pool["quantity"], 3),
                    round(s104_pool["total_cost"], 2),
                )

                s104_pool["quantity"] += transactions.at[idx, "unallocated_qty"]
                s104_pool["total_cost"] += (
                    transactions.at[idx, "txn_amount_gbp"]
                    / transactions.at[idx, "quantity"]
                    * transactions.at[idx, "unallocated_qty"]
                )

                transactions.at[idx, "unallocated_qty"] = 0

                transactions.at[idx, "s104_post_txn"] = (
                    round(s104_pool["quantity"], 3),
                    round(s104_pool["total_cost"], 2),
                )
        else:  # sell transaction
            # Check bed-and-breakfast rules
            # All allocations for same-day rule have completed by now.
            future_buys = transactions[
                (transactions["txn_type"] == "buy")
                & (transactions["date"] > transactions.at[idx, "date"])
                & ((transactions["date"] - transactions.at[idx, "date"]).dt.days <= 30)
            ]

            for buy_idx, _ in future_buys.iterrows():
                if transactions.at[idx, "unallocated_qty"] <= 0:
                    break

                if transactions.at[buy_idx, "unallocated_qty"] <= 0:
                    continue

                match_qty = round(
                    min(
                        transactions.at[idx, "unallocated_qty"],
                        transactions.at[buy_idx, "unallocated_qty"],
                    ),
                    3,
                )

                if match_qty > 0:  # avoid float precision error
                    transactions.at[idx, "allocations"].append(
                        {
                            "rule": "b-n-b",
                            "matched_quantity": match_qty,
                            "matched_cost": round(
                                transactions.at[buy_idx, "txn_amount_gbp"]
                                / transactions.at[buy_idx, "quantity"]
                                * match_qty,
                                2,
                            ),
                            "matched_txnid": int(transactions.at[buy_idx, "txnidx"]),
                        }
                    )

                    transactions.at[buy_idx, "allocations"].append(
                        {
                            "rule": "b-n-b",
                            "matched_quantity": match_qty,
                            "matched_txnid": int(transactions.at[idx, "txnidx"]),
                        }
                    )

                    transactions.at[idx, "unallocated_qty"] -= match_qty
                    transactions.at[buy_idx, "unallocated_qty"] -= match_qty

            transactions.at[idx, "s104_pre_txn"] = (
                round(s104_pool["quantity"], 3),
                round(s104_pool["total_cost"], 2),
            )

            # if unmatched qty remains, use s104
            if round(transactions.at[idx, "unallocated_qty"], 3) > 0:
                transactions.at[idx, "allocations"].append(
                    {
                        "rule": "s104",
                        "matched_quantity": round(
                            transactions.at[idx, "unallocated_qty"], 3
                        ),
                        "matched_cost": round(
                            transactions.at[idx, "unallocated_qty"]
                            * (s104_pool["total_cost"] / s104_pool["quantity"]),
                            2,
                        ),
                    }
                )

                s104_avg_cost = s104_pool["total_cost"] / s104_pool["quantity"]
                s104_pool["quantity"] -= transactions.at[idx, "unallocated_qty"]
                s104_pool["total_cost"] -= (
                    transactions.at[idx, "unallocated_qty"] * s104_avg_cost
                )

                transactions.at[idx, "unallocated_qty"] = 0

            transactions.at[idx, "s104_post_txn"] = (
                round(s104_pool["quantity"], 3),
                round(s104_pool["total_cost"], 2),
            )

            # compute P&L
            total_cost = sum(
                [alloc["matched_cost"] for alloc in transactions.at[idx, "allocations"]]
            )
            sale_proceeds = transactions.at[idx, "txn_amount_gbp"]
            transactions.at[idx, "capital_pnl"] = round(sale_proceeds - total_cost, 2)
            transactions.at[idx, "note"] = (
                f"Capital P&L = Sale Proceeds ({sale_proceeds:.2f}) - "
                f"Total Cost ({total_cost:.2f})"
            )

    transactions["allocations_note"] = transactions["allocations"].apply(
        convert_allocations_to_note
    )

    return transactions


# Convert postings to transaction representation
transactions_df = transform_to_transaction_representation(postings_df)
transactions_df = transactions_df.sort_values(by="date")

# Stock split adjustment
split_transaction_date = pd.Timestamp("2022-07-15")
split_transaction_mask = transactions_df["date"] == split_transaction_date
transactions_df.loc[split_transaction_mask, "note"] = (
    "Stock split adjustment: 20-for-1 split on July 15, 2022"
)
transactions_df.loc[split_transaction_mask, "txn_type"] = "split"

# Filter out transactions for stocks without any sell in the tax reporting
# period.
start_date = pd.Timestamp("2023-04-06")
end_date = pd.Timestamp("2024-04-05")

# Filter out stock symbols with no 'sell' transactions within the date range
relevant_stock_symbols = transactions_df[
    (transactions_df["txn_type"] == "sell")
    & (transactions_df["date"] >= start_date)
    & (transactions_df["date"] <= end_date)
]["stock_symbol"].unique()

# Retain only transactions for relevant stock symbols
transactions_df = transactions_df[
    transactions_df["stock_symbol"].isin(relevant_stock_symbols)
]

columns_to_keep = [
    "txnidx",
    "date",
    "txn_type",
    "s104_pre_txn",
    "quantity",
    "txn_unit_price_gbp",
    "txn_amount_gbp",
    "allocations_note",
    "s104_post_txn",
    "capital_pnl",
    "note",
]

# Use a dictionary to store results keyed by stock symbol
results_by_stock_symbol = {}

for stock_symbol, group in transactions_df.groupby("stock_symbol"):
    augmented_group = compute_capital_gains_with_pool(group)
    augmented_group["date"] = augmented_group["date"].dt.date
    results_by_stock_symbol[stock_symbol] = augmented_group[
        columns_to_keep
    ]  # augmented_group

transactions_df["date"] = transactions_df["date"].dt.date

# Final reporting
with pd.ExcelWriter("stock_transactions_report.xlsx", engine="openpyxl") as writer:
    # Write the 'All Transactions' sheet
    transactions_df.to_excel(writer, sheet_name="All Transactions", index=False)

    # Write individual sheets for each stock symbol
    for stock_symbol, df in results_by_stock_symbol.items():
        # Sanitize the sheet name to ensure it's a valid Excel sheet name
        valid_sheet_name = stock_symbol[
            :31
        ]  # Excel sheet names must be <= 31 characters
        df.to_excel(writer, sheet_name=valid_sheet_name, index=False)
