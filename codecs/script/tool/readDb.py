#!/usr/bin/env python3
import sqlite3
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Read encoding result SQLite database and display/export")
    parser.add_argument("--db", type=str, default="results.db",
                        help="SQLite database file path")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional: Path to export as CSV file")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1) Connect to SQLite database
    conn = sqlite3.connect(args.db)

    # 2) Use pandas to directly read the entire table
    df = pd.read_sql_query("SELECT * FROM results", conn)

    # 3) Close connection
    conn.close()

    # 4) Print to terminal
    print(df.to_string(index=False))

    # 5) If --csv is specified, export
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nResults exported to: {args.csv}")

if __name__ == "__main__":
    main()