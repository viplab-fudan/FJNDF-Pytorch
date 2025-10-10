#!/usr/bin/env python3
"""
calc_bpp_saving.py

Compare two SQLite result databases (anchor.db / test.db):
1) Bit rate reduction percentage for each video, each QP
2) Average bit rate reduction percentage for all videos under the same QP
"""

import argparse
import sqlite3
import pandas as pd
import numpy as np
import sys

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(
        description="Calculate bit rate reduction percentage between two SQLite result databases")
    p.add_argument(
        "--anchor", "-a", required=True,
        help="anchor database file (SQLite)")
    p.add_argument(
        "--test", "-t", required=True,
        help="test database file (SQLite)")
    p.add_argument(
        "--out-csv", "-o", default=None,
        help="If specified, write results to CSV file")
    return p.parse_args()

# ---------- core functions ----------
def find_valid_table(conn: sqlite3.Connection) -> str:
    """
    Find table containing 'video', 'enc_ratio', 'bpp' three columns in given connection:
    1. List all user tables from sqlite_master
    2. Execute PRAGMA table_info for each table, check if it contains required columns
    3. If find unique match, return it; if multiple matches, return first and print warning; if no match, throw error
    """
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [row[0] for row in cur.fetchall()]
    valid_tables = []
    for tbl in tables:
        cur.execute(f"PRAGMA table_info('{tbl}')")
        cols = [r[1] for r in cur.fetchall()]  # PRAGMA table_info: (cid, name, type, ...)
        if all(col in cols for col in ('video', 'enc_ratio', 'bpp')):
            valid_tables.append(tbl)

    if not valid_tables:
        raise KeyError(
            f"Could not find table containing columns 'video', 'enc_ratio', 'bpp' in database. Available tables: {tables}"
        )
    if len(valid_tables) > 1:
        print(
                    f"Warning: Found multiple tables containing 'video', 'enc_ratio', 'bpp',"
        f"default to first: '{valid_tables[0]}'", file=sys.stderr
        )
    return valid_tables[0]

def read_db(db_path: str) -> pd.DataFrame:
    """
    Read table containing columns 'video', 'enc_ratio', 'bpp' from SQLite file, return corresponding DataFrame
    """
    conn = sqlite3.connect(db_path)
    try:
        table = find_valid_table(conn)
        df = pd.read_sql_query(f"SELECT video, enc_ratio, bpp FROM '{table}'", conn)
    finally:
        conn.close()
    return df

def compute_per_video_saving(a_df: pd.DataFrame,
                             b_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate bit rate reduction percentage for each video, each QP
    Returns DataFrame with columns ['video', 'enc_ratio', 'bpp_anchor', 'bpp_test', 'saving_%']
    """
    videos = sorted(set(a_df['video']) & set(b_df['video']))
    rows   = []
    for vid in videos:
        aa = a_df[a_df['video'] == vid]
        bb = b_df[b_df['video'] == vid]

        # Require both sides to be completely consistent across all QPs
        if not np.array_equal(
            np.sort(aa['enc_ratio'].values),
            np.sort(bb['enc_ratio'].values)
        ):
            raise ValueError(f"{vid}: anchor/test QP mismatch")

        merged = aa[['enc_ratio', 'bpp']].merge(
                 bb[['enc_ratio', 'bpp']],
                 on='enc_ratio',
                 suffixes=('_anchor', '_test'))

        merged['saving_%'] = (
            (merged['bpp_anchor'] - merged['bpp_test'])
            / merged['bpp_anchor'] * 100
        )
        merged.insert(0, 'video', vid)
        rows.append(merged)

    return pd.concat(rows, ignore_index=True)

def compute_avg_saving(per_video_df: pd.DataFrame) -> pd.DataFrame:
    """
    Based on per_video_df, calculate average bpp_anchor, bpp_test, saving_% for all videos by QP (enc_ratio),
    and rename 'saving_%' to 'saving_%_mean'
    Returns DataFrame with columns ['enc_ratio', 'bpp_anchor', 'bpp_test', 'saving_%_mean']
    """
    grp = per_video_df.groupby('enc_ratio', as_index=False)
    avg = grp.agg({
        'bpp_anchor': 'mean',
        'bpp_test'  : 'mean',
        'saving_%'  : 'mean'
    })
    avg.rename(columns={'saving_%': 'saving_%_mean'}, inplace=True)
    return avg

# ---------- main ----------
def main():
    args = parse_args()
    anchor = read_db(args.anchor)
    test   = read_db(args.test)

    # The three most basic columns must exist
    for col in ('video', 'enc_ratio', 'bpp'):
        if col not in anchor.columns or col not in test.columns:
            raise KeyError(f"Missing column {col}")

    # Calculate per-video and average results
    per_video = compute_per_video_saving(anchor, test)
    avg_qp    = compute_avg_saving(per_video)

    # Print to console
    print("\n=== Per-video saving (%) ===")
    print(
        per_video.to_string(
            index=False,
            formatters={'saving_%': '{:.2f}'.format}
        )
    )

    print("\n=== Average saving across videos ===")
    print(
        avg_qp.to_string(
            index=False,
            formatters={'saving_%_mean': '{:.2f}'.format}
        )
    )

    # If --out-csv is specified, write per-video results and avg_qp results to the same CSV file
    if args.out_csv:
        # Convert avg_qp to same format as per_video, set "video" column to 'ALL'
        avg_df = avg_qp.copy()
        avg_df['video'] = 'ALL'
        # Rename column 'saving_%_mean' to 'saving_%'
        avg_df.rename(columns={'saving_%_mean': 'saving_%'}, inplace=True)
        # Rearrange according to per_video column order
        avg_df = avg_df[['video', 'enc_ratio', 'bpp_anchor', 'bpp_test', 'saving_%']]

        # Concatenate per_video and avg_df
        combined = pd.concat([per_video, avg_df], ignore_index=True)

        # Write to CSV
        combined.to_csv(args.out_csv, index=False, float_format='%.2f')
        print(f"\nPer-video and average results saved to {args.out_csv}")

if __name__ == "__main__":
    main()