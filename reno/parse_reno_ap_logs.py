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


def empty_row(rate_id, scale='', posq=''):
    return {
        'rate_id': rate_id,
        'scale_label': scale,
        'posQ': posq,
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
            rate_match = re.search(r'Start RENO Evaluation rate_id=(\d+)\s+scale=([^\s]+)\s+posQ=([0-9.eE+-]+)', line)
            if rate_match:
                if row is not None:
                    rows.append(row)
                row = empty_row(int(rate_match.group(1)), rate_match.group(2), rate_match.group(3))
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


def parse_one_log(path, rate_id, posq):
    row = empty_row(rate_id, '', posq)
    waiting_class = None
    with open(path, encoding='utf-8', errors='replace') as f:
        for raw_line in f:
            line = raw_line.strip()
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
    return row


def parse_posqs(value):
    return [int(x) for x in str(value).replace(',', ' ').split() if x.strip()]


def main():
    parser = argparse.ArgumentParser(description='Parse one AP log per RENO posQ into a CSV.')
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--posqs', default='4,8,16,32,64,128,256,512')
    parser.add_argument('--combined_log', default='', help='Optional single test_reno_pos.py log containing all rates.')
    parser.add_argument('--log_pattern', default='ap_posQ_{posQ}.log')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if args.combined_log:
        rows = parse_combined_log(args.combined_log)
    else:
        rows = []
        for rate_id, posq in enumerate(parse_posqs(args.posqs)):
            log_path = log_dir / args.log_pattern.format(posQ=posq, rate_id=rate_id)
            if not log_path.exists():
                raise FileNotFoundError(log_path)
            rows.append(parse_one_log(log_path, rate_id, posq))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Parsed {len(rows)} RENO AP rows -> {args.out}')


if __name__ == '__main__':
    main()
