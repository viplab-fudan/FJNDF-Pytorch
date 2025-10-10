#!/usr/bin/env python3
"""
process_konjnd_1k.py

Restructure the Konjnd1K dataset directory by copying the "bpg" and "jpeg" folders
into a "reference_images" directory, and the "source_image" folder into a "input_images" directory
in the target location. Additionally, convert the original subjective_ratings.csv into a meta-info CSV
with columns: input_image, reference_image, jnd_level, number_of_ratings, jnd_mean, jnd_std, jnd_ratings.

python process_konjnd_1k.py --src_dir ../datasets/konjnd_1k \
                            --dst_dir ../datasets/konjnd_1k_tar \
                            --input_csv ../datasets/konjnd_1k/subjective_ratings.csv \
                            --output_csv ../datasets/info/jnd/konjnd_1k_meta_info.csv
"""
import os
import shutil
import argparse
import csv


def is_image_file(filename):
    IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')
    return filename.lower().endswith(IMG_EXTENSIONS)


def copy_files(src_root, subdirs, dst_root):
    """
    Copy image files from specified subdirectories under src_root into dst_root.
    Preserve relative directory structure.
    """
    count = 0
    for sub in subdirs:
        src_path = os.path.join(src_root, sub)
        if not os.path.isdir(src_path):
            print(f"Warning: source subdirectory {src_path} does not exist, skipping.")
            continue
        for root, _, files in os.walk(src_path):
            rel_path = os.path.relpath(root, src_path)
            dst_dir = os.path.join(dst_root, rel_path) if rel_path != '.' else dst_root
            os.makedirs(dst_dir, exist_ok=True)
            for fname in files:
                if is_image_file(fname):
                    src_file = os.path.join(root, fname)
                    dst_file = os.path.join(dst_dir, fname)
                    shutil.copy2(src_file, dst_file)
                    count += 1
    return count


def restructure_dataset(src_dir, dst_dir):
    """
    Perform directory restructuring: copy bpg/jpeg to reference_images,
    source_image to input_images.
    """
    mapping = {
        'bpg': 'reference_images',
        'jpeg': 'reference_images',
        'source_image': 'input_images'
    }
    for target_sub in set(mapping.values()):
        os.makedirs(os.path.join(dst_dir, target_sub), exist_ok=True)

    total_copied = 0
    for orig_sub, new_sub in mapping.items():
        print(f"Processing '{orig_sub}' -> '{new_sub}'...")
        dst_sub_path = os.path.join(dst_dir, new_sub)
        copied = copy_files(src_dir, [orig_sub], dst_sub_path)
        print(f"  Copied {copied} files from '{orig_sub}' to '{new_sub}'.")
        total_copied += copied

    print(f"Finished restructuring. Total files copied: {total_copied}.")


def convert_csv(input_csv, output_csv):
    """
    Convert the original subjective_ratings.csv into a meta-info CSV with fields:
    input_image, reference_image, jnd_level, number_of_ratings, jnd_mean, jnd_std, jnd_ratings
    """
    if not os.path.isfile(input_csv):
        print(f"Error: CSV input file '{input_csv}' not found.")
        return

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(input_csv, newline='', encoding='utf-8') as csvfile, \
         open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(csvfile)
        fieldnames = ['input_image', 'reference_image', 'jnd_level',
                      'number_of_ratings', 'jnd_mean', 'jnd_std', 'jnd_ratings']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in reader:
            image_id = row['image_id']
            comp = row['Compression type']
            num_ratings = row['No. of ratings']
            mean_val = float(row['mean'])
            std_val = float(row['std'])
            ratings = row['ratings']

            base = os.path.splitext(image_id)[0]
            comp_upper = comp.upper()
            if comp_upper == 'JPEG':
                level = round(mean_val)
                ext = '.jpg'
            elif comp_upper == 'BPG':
                level = round(mean_val / 2)
                ext = '.png'
            else:
                print(f"Warning: Unknown compression type '{comp}' for '{image_id}', skipping.")
                continue

            distorted_name = f"{base}_{comp_upper}_{level:03d}{ext}"
            input_path = os.path.join('input_images', image_id)
            ref_path = os.path.join('reference_images', distorted_name)
            jnd_level = 1

            writer.writerow({
                'input_image': input_path,
                'reference_image': ref_path,
                'jnd_level': jnd_level,
                'number_of_ratings': num_ratings,
                'jnd_mean': mean_val,
                'jnd_std': std_val,
                'jnd_ratings': ratings
            })

    print(f"Converted CSV written to '{output_csv}'")


def check_distorted(src_dir, dst_dir):
    """
    Verify all image files under src_dir/bpg and src_dir/jpeg exist in dst_dir/reference_images.
    Prints missing files and summary.
    """
    src_paths = []
    for sub in ('bpg', 'jpeg'):
        root_path = os.path.join(src_dir, sub)
        for root, _, files in os.walk(root_path):
            for f in files:
                if is_image_file(f):
                    src_paths.append(os.path.relpath(os.path.join(root, f), root_path))

    dst_root = os.path.join(dst_dir, 'reference_images')
    dst_files = set()
    for root, _, files in os.walk(dst_root):
        for f in files:
            if is_image_file(f):
                dst_files.add(os.path.relpath(os.path.join(root, f), dst_root))

    missing = []
    for rel in src_paths:
        if rel not in dst_files:
            missing.append(rel)

    if missing:
        print(f"Missing {len(missing)} files in '{dst_root}':")
        for m in missing:
            print(f"  {m}")
    else:
        print(f"All {len(src_paths)} source files were found in reference_images.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Restructure Konjnd1K dataset and optionally verify reference_images completeness.'
    )
    parser.add_argument('--src_dir', help='Original konjnd_1k directory')
    parser.add_argument('--dst_dir', help='Target konjnd_1k_tar directory')
    parser.add_argument('--input_csv',
                        help='Path to subjective_ratings.csv for meta CSV conversion')
    parser.add_argument('--output_csv',
                        help='Path where konjnd_1k_meta_info.csv will be written')
    parser.add_argument('--check', action='store_true',
                        help='Check that all bpg/jpeg images were copied to reference_images')
    args = parser.parse_args()

    if not os.path.isdir(args.src_dir):
        print(f"Error: source directory '{args.src_dir}' does not exist.")
        exit(1)
    os.makedirs(args.dst_dir, exist_ok=True)

    restructure_dataset(args.src_dir, args.dst_dir)

    if args.input_csv and args.output_csv:
        convert_csv(args.input_csv, args.output_csv)

    if args.check:
        check_distorted(args.src_dir, args.dst_dir)