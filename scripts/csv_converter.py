#!/usr/bin/env python3
"""
csv_converter.py

Convert input CSV (headers: filename,resolution,bitdepth,format,number)
to output CSV (headers: input_image,reference_image,yuv_width,
yuv_height,yuv_format,yuv_bitdepth,jnd_mean)

Usage:
    python csv_converter.py input.csv output.csv
"""

import csv
import argparse

def convert_csv(input_path: str, output_path: str):
    with open(input_path, newline='', encoding='utf-8') as inf, \
         open(output_path, 'w', newline='', encoding='utf-8') as outf:
        reader = csv.DictReader(inf)
        fieldnames = [
            'input_image',
            'reference_image',
            'yuv_width',
            'yuv_height',
            'yuv_format',
            'yuv_bitdepth',
            'jnd_mean'
        ]
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            filename = row.get('filename', '')
            resolution = row.get('resolution', '')
            # Split resolution into width and height
            if 'x' in resolution:
                yuv_width, yuv_height = resolution.split('x', 1)
            else:
                yuv_width = ''
                yuv_height = ''

            out_row = {
                'input_image': filename,
                'reference_image': filename,
                'yuv_width': yuv_width,
                'yuv_height': yuv_height,
                'yuv_format': row.get('format', ''),
                'yuv_bitdepth': row.get('bitdepth', ''),
                'jnd_mean': 0
            }
            writer.writerow(out_row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert CSV file format for YUV metadata and add jnd_mean column'
    )
    parser.add_argument('input_csv', help='Input CSV file path')
    parser.add_argument('output_csv', help='Output CSV file path')
    args = parser.parse_args()

    convert_csv(args.input_csv, args.output_csv)
    print(f"Conversion completed: {args.input_csv} → {args.output_csv}")
