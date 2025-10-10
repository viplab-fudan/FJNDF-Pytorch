#!/usr/bin/env python3
"""
This script automatically detects all CSV files in a given root directory.
Each CSV file should be named following the pattern: FILTER_METHOD_ENCODER.csv.
Each CSV must have the following columns:
  video, metric, BD-Rate(%), BD-PSNR(dB)

The script processes rows where the 'video' column is 'ALL' to get the average results.
The methods are sorted by their average BD-Rate across all metrics (in the 'all' column).

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
    Groups results by encoder, transposes the data so methods are rows and
    metrics are columns, adds an 'all' column with the average BD-Rate,
    sorts methods by the 'all' column, and prints a Markdown table.
    """
    # Define the desired custom order for metrics (columns) and encoders (tables)
    metric_order = ['psnr', 'psnr_hvsm', 'ssim', 'ms_ssim', 'vmaf', 'vmaf_neg']
    encoder_order = ['x264', 'x265', 'aomenc', 'vvenc']

    # --- Loop through the predefined encoder order ---
    for encoder in encoder_order:
        # Select data for the current encoder
        enc_df = df[df['encoder'] == encoder]

        # If no data exists for this encoder, skip it
        if enc_df.empty:
            continue

        print(f"## {encoder}\n")

        # Methods are now rows (index), metrics are columns
        pivot = enc_df.pivot_table(
            index='method',
            columns='metric',
            values='bd_rate'
        )
        
        present_metrics_in_order = [m for m in metric_order if m in pivot.columns]
        other_metrics = sorted([m for m in pivot.columns if m not in present_metrics_in_order])
        final_metric_order = present_metrics_in_order + other_metrics
        pivot = pivot.reindex(final_metric_order, axis=1)

        # Ensure the table is not empty before proceeding
        if not pivot.empty:
            pivot['all'] = pivot.mean(axis=1)
            
            pivot = pivot.sort_values(by='all', ascending=True)
        
        # Print the final table in Markdown format
        print(pivot.to_markdown(floatfmt=".2f"))
        print("\n")


def main():
    parser = argparse.ArgumentParser(
        description='Parse CSV results and output transposed Markdown tables of BD-Rate, sorted by average performance.'
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