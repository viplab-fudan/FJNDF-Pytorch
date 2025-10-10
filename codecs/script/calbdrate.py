#!/usr/bin/env python3
"""
compute_bd.py

Calculate BD-Rate / BD-PSNR between two encoding result SQLite databases:
1) Per-video (calculate each video separately)
2) Across-videos average (first average all videos under same QP, then calculate BD of average curve)
"""

import argparse
import sqlite3
import pandas as pd
import numpy as np
import bjontegaard as bd

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(
        description="Calculate BD-Rate / BD-PSNR between two encoding result SQLite databases")
    p.add_argument(
        "--anchor", "-a", required=True,
        help="anchor database file (SQLite)")
    p.add_argument(
        "--test", "-t", required=True,
        help="test database file (SQLite)")
    p.add_argument(
        "--out-csv", "-o", default=None,
        help="If specified, write per-video and average results to CSV file")
    return p.parse_args()

# ---------- core functions ----------
def read_db(db_path: str) -> pd.DataFrame:
    """
    Use pandas to read entire results table from SQLite, return DataFrame.
Assume table name is 'results', and contains at least ['video', 'enc_ratio', 'bpp', <metrics...>] columns.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM results", conn)
    conn.close()
    return df

def compute_per_video_bd(
    anchor_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics: list
) -> pd.DataFrame:
    """Calculate BD-Rate/BD-PSNR for each video separately, remove nan/inf points before calculation, return nan if points <2"""
    videos = sorted(set(anchor_df['video']) & set(test_df['video']))
    records = []
    for vid in videos:
        a = anchor_df[anchor_df['video'] == vid].sort_values('enc_ratio')
        b = test_df  [test_df  ['video'] == vid].sort_values('enc_ratio')
        if not np.array_equal(a['enc_ratio'].values, b['enc_ratio'].values):
            raise ValueError(f"Video {vid}: anchor/test QP mismatch!")
        R1_full = a['bpp'].values
        R2_full = b['bpp'].values

        for m in metrics:
            Q1_full = a[m].values
            Q2_full = b[m].values

            # Construct mask, remove nan/inf
            mask = (
                ~np.isnan(R1_full) & ~np.isnan(R2_full) &
                ~np.isnan(Q1_full) & ~np.isnan(Q2_full) &
                ~np.isinf(R1_full) & ~np.isinf(R2_full) &
                ~np.isinf(Q1_full) & ~np.isinf(Q2_full)
            )
            R1 = R1_full[mask]
            R2 = R2_full[mask]
            Q1 = Q1_full[mask]
            Q2 = Q2_full[mask]

            # Need at least four points to calculate
            if len(R1) < 4:
                bd_rate = np.nan
                bd_psnr = np.nan
            else:
                try:
                    bd_rate = bd.bd_rate(R1, Q1, R2, Q2, method='akima')
                except Exception:
                    bd_rate = np.nan
                try:
                    bd_psnr = bd.bd_psnr(R1, Q1, R2, Q2, method='akima')
                except Exception:
                    bd_psnr = np.nan

            records.append({
                'video': vid,
                'metric': m,
                'BD-Rate(%)': bd_rate,
                'BD-PSNR(dB)': bd_psnr
            })
    return pd.DataFrame.from_records(records)


def compute_average_bd(
    anchor_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics: list
) -> pd.DataFrame:
    """Process each metric separately: remove videos with nan/inf then average, then calculate BD"""
    records = []
    for m in metrics:
        # Only keep three columns to avoid interference from irrelevant columns
        a = anchor_df[['enc_ratio', 'bpp', m]].copy()
        b = test_df[['enc_ratio', 'bpp', m]].copy()

        # Convert inf to nan
        a.replace([np.inf, -np.inf], np.nan, inplace=True)
        b.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Remove rows where metric or bpp is nan
        a_clean = a.dropna(subset=['bpp', m])
        b_clean = b.dropna(subset=['bpp', m])

        # Group by enc_ratio and calculate mean
        a_grp = a_clean.groupby('enc_ratio').mean().reset_index().sort_values('enc_ratio')
        b_grp = b_clean.groupby('enc_ratio').mean().reset_index().sort_values('enc_ratio')

        # Only keep enc_ratio points that exist on both sides
        common = np.intersect1d(a_grp['enc_ratio'], b_grp['enc_ratio'])
        a_avg = a_grp[a_grp['enc_ratio'].isin(common)]
        b_avg = b_grp[b_grp['enc_ratio'].isin(common)]

        R1 = a_avg['bpp'].values
        R2 = b_avg['bpp'].values
        Q1 = a_avg[m].values
        Q2 = b_avg[m].values

        if len(R1) < 4:
            bd_rate = np.nan
            bd_psnr = np.nan
        else:
            try:
                bd_rate = bd.bd_rate(R1, Q1, R2, Q2, method='akima')
            except Exception:
                bd_rate = np.nan
            try:
                bd_psnr = bd.bd_psnr(R1, Q1, R2, Q2, method='akima')
            except Exception:
                bd_psnr = np.nan

        records.append({
            'metric': m,
            'BD-Rate(%)': bd_rate,
            'BD-PSNR(dB)': bd_psnr
        })
    return pd.DataFrame.from_records(records)

# ---------- main ----------
def main():
    args = parse_args()
    # 1) Read data from SQLite
    anchor_df = read_db(args.anchor)
    test_df   = read_db(args.test)

    # 2) Validate required columns
    #    Besides ['video','enc_ratio','bpp'], also need metrics columns for subsequent BD calculation
#    Assume results table has other columns like 'PSNR', 'SSIM', etc. Here only example 'PSNR'
    metrics = [col for col in anchor_df.columns if col not in ('video', 'enc_ratio', 'bpp', 'ssimulacra2')]
    required_cols = ['video', 'enc_ratio', 'bpp'] + metrics
    for col in required_cols:
        if col not in anchor_df.columns or col not in test_df.columns:
            raise KeyError(f"Missing required column `{col}`, please check database schema.")

    # 3) Per-video BD calculation
    per_video_df = compute_per_video_bd(anchor_df, test_df, metrics)

    # 4) Average result BD calculation
    avg_df = compute_average_bd(anchor_df, test_df, metrics)

    # 5) Display
    print("\n=== Per-video BD results ===")
    print(per_video_df.to_string(index=False))

    print("\n=== Average (all videos) BD results ===")
    print(avg_df.to_string(index=False))

    # 6) Optional: Export per-video and avg CSV
    if args.out_csv:
        # 6.1) Add 'video' column to avg_df and set to 'ALL'
        avg_out = avg_df.copy()
        avg_out['video'] = 'ALL'

        # 6.2) Rearrange column order to match per_video_df
        # per_video_df columns are ['video','metric','BD-Rate(%)','BD-PSNR(dB)']
        avg_out = avg_out[['video', 'metric', 'BD-Rate(%)', 'BD-PSNR(dB)']]

        # 6.3) Concatenate per_video_df and avg_out
        combined = pd.concat([per_video_df, avg_out], ignore_index=True)

        # 6.4) Write to same CSV file
        combined.to_csv(args.out_csv, index=False)
        print(f"\nPer-video and Average results saved to: {args.out_csv}")

if __name__ == "__main__":
    main()
