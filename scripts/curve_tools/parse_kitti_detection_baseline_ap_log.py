import argparse
import csv
import re


CLASS_HEADERS = {
    'Car': 'Car AP_R40@0.70, 0.70, 0.70',
    'Pedestrian': 'Pedestrian AP_R40@0.50, 0.50, 0.50',
    'Cyclist': 'Cyclist AP_R40@0.50, 0.50, 0.50',
}


def parse_log(log_path):
    rows = []
    current = None
    waiting_class = None

    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for raw_line in f:
            line = raw_line.strip()

            scale_match = re.search(r'Start Evaluation for Scale:\s*([0-9.eE+-]+)', line)
            if scale_match:
                if current is not None:
                    rows.append(current)
                scale = float(scale_match.group(1))
                current = {
                    'rate_id': len(rows),
                    'scale': scale,
                    'posQuantscale': scale,
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
                waiting_class = None
                continue

            if current is None:
                continue

            for cls_name, header in CLASS_HEADERS.items():
                if header in line:
                    waiting_class = cls_name
                    break
            else:
                if waiting_class and line.startswith('3d') and 'AP:' in line:
                    ap_part = line.split('AP:', 1)[1]
                    values = [float(x) for x in re.findall(r'[0-9]+(?:\.[0-9]+)?', ap_part)]
                    if len(values) >= 3:
                        current[f'{waiting_class}_3d_AP_R40_easy'] = values[0]
                        current[f'{waiting_class}_3d_AP_R40_moderate'] = values[1]
                        current[f'{waiting_class}_3d_AP_R40_hard'] = values[2]
                    waiting_class = None

    if current is not None:
        rows.append(current)

    return rows


def main():
    parser = argparse.ArgumentParser(description='Parse KITTI fixed-quantization AP values from evaluator logs.')
    parser.add_argument('--log', required=True, help='Path to log_eval_pos_*.txt')
    parser.add_argument('--out', required=True, help='Output AP CSV')
    args = parser.parse_args()

    rows = parse_log(args.log)
    if not rows:
        raise RuntimeError(f'No AP rows parsed from {args.log}')

    fieldnames = list(rows[0].keys())
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Parsed {len(rows)} AP rows -> {args.out}')


if __name__ == '__main__':
    main()
