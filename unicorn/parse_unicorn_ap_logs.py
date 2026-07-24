#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


CLASS_HEADERS = {
    'Car': 'Car AP_R40@0.70, 0.70, 0.70',
    'Pedestrian': 'Pedestrian AP_R40@0.50, 0.50, 0.50',
    'Cyclist': 'Cyclist AP_R40@0.50, 0.50, 0.50',
}


def empty_row(rate_id):
    return {
        'rate_id': rate_id,
        'Car_3d_AP_R40_easy': '',
        'Car_3d_AP_R40_moderate': '',
        'Car_3d_AP_R40_hard': '',
        'Pedestrian_3d_AP_R40_easy': '',
        'Pedestrian_3d_AP_R40_moderate': '',
        'Pedestrian_3d_AP_R40_hard': '',
        'Cyclist_3d_AP_R40_easy': '',
        'Cyclist_3d_AP_R40_moderate': '',
        'Cyclist_3d_AP_R40_hard': '',
    }


def parse_combined_log(path):
    rows = []
    row = None
    waiting_class = None
    with open(path, encoding='utf-8', errors='replace') as f:
        for raw_line in f:
            line = raw_line.strip()
            rate_match = re.search(r'Start Unicorn Evaluation rate_id=(\d+)', line)
            if rate_match:
                if row is not None:
                    rows.append(row)
                row = empty_row(int(rate_match.group(1)))
                waiting_class = None
                continue
            if row is None:
                continue
            for cls_name, header in CLASS_HEADERS.items():
                if header in line:
                    waiting_class = cls_name
                    break
            else:
                if waiting_class and line.startswith('3d') and 'AP:' in line:
                    values = [float(x) for x in re.findall(r'[0-9]+(?:\.[0-9]+)?', line.split('AP:', 1)[1])]
                    if len(values) >= 3:
                        row[f'{waiting_class}_3d_AP_R40_easy'] = values[0]
                        row[f'{waiting_class}_3d_AP_R40_moderate'] = values[1]
                        row[f'{waiting_class}_3d_AP_R40_hard'] = values[2]
                    waiting_class = None
    if row is not None:
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description='Parse Unicorn AP log into a CSV.')
    parser.add_argument('--combined_log', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    rows = parse_combined_log(args.combined_log)
    if not rows:
        raise RuntimeError(f'No Unicorn AP rows parsed from {args.combined_log}')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Parsed {len(rows)} Unicorn AP rows -> {args.out}')


if __name__ == '__main__':
    main()
