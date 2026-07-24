#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def read_by_rate(path):
    with open(path, newline='') as f:
        return {int(row['rate_id']): row for row in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser(description='Merge Unicorn rate/time/PSNR CSV and AP CSV.')
    parser.add_argument('--rate_csv', required=True)
    parser.add_argument('--ap_csv', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    rate_rows = read_by_rate(args.rate_csv)
    ap_rows = read_by_rate(args.ap_csv)
    rows = []
    for rate_id in sorted(rate_rows):
        if rate_id not in ap_rows:
            continue
        row = {}
        row.update(rate_rows[rate_id])
        for key, value in ap_rows[rate_id].items():
            if key not in row:
                row[key] = value
        rows.append(row)
    if not rows:
        raise RuntimeError('No matching Unicorn rows found between rate and AP CSVs.')

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Merged {len(rows)} Unicorn curve rows -> {args.out}')


if __name__ == '__main__':
    main()
