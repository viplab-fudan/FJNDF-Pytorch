#!/usr/bin/env python3
"""
Script to generate random train/val/test splits, output pkl file that can be parsed by BaseJNDDataset.

Usage:
  python csv_to_random_split_pkl.py \
      --csv input.csv \
      --output splits.pkl \
      [--splits 10] \
      [--seed 42] \
      [--train_ratio 0.8] [--val_ratio 0.1] [--test_ratio 0.1]

Requirements: Input CSV must contain headers:
  input_image,reference_image,yuv_width,yuv_height,yuv_format,yuv_bitdepth

Output pkl format:
  {1: {'train': [...], 'val': [...], 'test': [...]},
   2: {...},
   ...
   10: {...}}

Where index corresponds to CSV file row number (0-based).
"""
import argparse
import os
import pickle
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate random train/val/test splits and save to pickle.")
    parser.add_argument(
        "--csv", required=True,
        help="Path to input CSV file with specified header.")
    parser.add_argument(
        "--output", required=True,
        help="Path to output pickle file for splits.")
    parser.add_argument(
        "--splits", type=int, default=10,
        help="Number of random splits to generate (default: 10).")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility. If omitted, uses a random seed.")
    parser.add_argument(
        "--train_ratio", type=float, default=0.8,
        help="Ratio of samples for training set (default: 0.8).")
    parser.add_argument(
        "--val_ratio", type=float, default=0.1,
        help="Ratio of samples for validation set (default: 0.1).")
    parser.add_argument(
        "--test_ratio", type=float, default=0.1,
        help="Ratio of samples for test set (default: 0.1).")
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must sum to 1.0, got {total_ratio}")

    # Read CSV
    df = pd.read_csv(args.csv)
    n_samples = len(df)
    if n_samples == 0:
        raise ValueError("Input CSV file has no data rows.")

    # If seed not specified, generate a random seed
    seed = args.seed if args.seed is not None else np.random.SeedSequence().entropy
    rng = np.random.default_rng(seed)

    split_dict = {}
    for split_idx in range(1, args.splits + 1):
        perm = rng.permutation(n_samples)
        n_train = int(np.floor(args.train_ratio * n_samples))
        n_val = int(np.floor(args.val_ratio * n_samples))
        # Remaining goes to test
        n_test = n_samples - n_train - n_val

        train_idx = perm[:n_train].tolist()
        val_idx = perm[n_train:n_train + n_val].tolist()
        test_idx = perm[n_train + n_val:].tolist()

        split_dict[split_idx] = {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx,
        }

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Write pickle
    with open(args.output, 'wb') as f:
        pickle.dump(split_dict, f)

    print(f"Successfully wrote {args.splits} splits to {args.output} (seed={int(seed)})")


if __name__ == '__main__':
    main()
