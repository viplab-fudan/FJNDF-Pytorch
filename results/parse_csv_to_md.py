#!/usr/bin/env python3
"""
This script automatically detects all CSV files in a given root directory.
Each CSV file should be named following the pattern: FILTER_METHOD_ENCODER.csv.
Each CSV must have the following columns:
  video, metric, BD-Rate(%), BD-PSNR(dB)

The script processes rows where the 'video' column is 'ALL' to get the average results.
It then extracts the BD-Rate for each filter-method/encoder/metric combination, rounds it
to two decimal places, and prints a Markdown table for each encoder in a predefined order.
An 'all' row showing the average BD-Rate across all metrics is added to each table.

Usage:
    python parse_csv_to_md.py /path/to/root_directory
"""
import os
import argparse
import pandas as pd

def collect_results(root_dir):
    """
    Collects and parses BD-Rate results from all CSV files in the root directory.
    """
    rows = []
    # Find all CSV files directly in the root directory
    csv_files = [f for f in os.listdir(root_dir) if f.lower().endswith('.csv')]

    if not csv_files:
        print(f"Warning: No CSV files found in {root_dir}")
        return pd.DataFrame()

    for fname in csv_files:
        # Parse method and encoder from the filename
        base = os.path.splitext(fname)[0]
        idx = base.rfind('_')
        if idx == -1:
            print(f"Warning: Skipping file with unexpected name format: {fname}")
            continue
            
        method = base[:idx]
        encoder = base[idx+1:]

        file_path = os.path.join(root_dir, fname)
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Warning: Cannot read {file_path}: {e}")
            continue

        # Filter for rows with average results ('ALL' videos)
        df_all = df[df['video'] == 'ALL']
        for _, row in df_all.iterrows():
            metric = row['metric']
            bd_val = str(row['BD-Rate(%)']).replace('%', '')
            try:
                bd_rate = float(bd_val)
            except (ValueError, TypeError):
                continue
                
            rows.append({
                'encoder': encoder,
                'metric': metric,
                'method': method,
                'bd_rate': round(bd_rate, 2)
            })
            
    return pd.DataFrame(rows)


def print_markdown_tables(df):
    """
    Groups results by a custom encoder order, adds an 'all' metric row,
    and prints a Markdown table for each encoder with a custom metric order.
    """
    # Define the desired custom order for metrics (rows) and encoders (tables)
    metric_order = ['psnr', 'psnr_hvsm', 'ssim', 'ms_ssim', 'vmaf', 'vmaf_neg']
    encoder_order = ['x264', 'x265', 'aomenc', 'vvenc']

    # --- NEW: Loop through the predefined encoder order ---
    for encoder in encoder_order:
        # Select data for the current encoder
        enc_df = df[df['encoder'] == encoder]

        # If no data exists for this encoder, skip it
        if enc_df.empty:
            continue

        print(f"## {encoder}\n")
        
        # Create a pivot table for the current encoder's data
        pivot = enc_df.pivot_table(
            index='metric',
            columns='method',
            values='bd_rate'
        )
        
        # Reorder the metric rows based on the custom list
        present_metrics_in_order = [m for m in metric_order if m in pivot.index]
        other_metrics = sorted([m for m in pivot.index if m not in present_metrics_in_order])
        final_metric_order = present_metrics_in_order + other_metrics
        pivot = pivot.reindex(final_metric_order).sort_index(axis=1)

        # Ensure the table is not empty before calculating the mean
        if not pivot.empty:
            # Calculate the mean of each column and add it as a new row named 'all'
            pivot.loc['all'] = pivot.mean()
        
        # Print the final table in Markdown format
        print(pivot.to_markdown(floatfmt=".2f"))
        print("\n")


def main():
    parser = argparse.ArgumentParser(
        description='Parse CSV results from a directory and output Markdown tables of BD-Rate, grouped by encoder.'
    )
    parser.add_argument('root', help='Root directory containing the result CSV files.')
    args = parser.parse_args()

    results_df = collect_results(args.root)
    if results_df.empty:
        print("No valid data found in the provided root directory.")
        return

    print_markdown_tables(results_df)

if __name__ == '__main__':
    main()