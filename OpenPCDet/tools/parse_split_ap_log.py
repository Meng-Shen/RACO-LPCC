import argparse
import csv
import re


CLASS_HEADERS = {
    'Car': r'Car\s+AP_R40@0\.70,\s*0\.70,\s*0\.70',
    'Pedestrian': r'Pedestrian\s+AP_R40@0\.50,\s*0\.50,\s*0\.50',
    'Cyclist': r'Cyclist\s+AP_R40@0\.50,\s*0\.50,\s*0\.50',
}


def empty_row(combo_id, fg_scale, bg_scale):
    return {
        'rate_id': combo_id,
        'combo_id': combo_id,
        'scale': f'{fg_scale},{bg_scale}',
        'posQuantscale': f'{fg_scale},{bg_scale}',
        'posQ_fg': fg_scale,
        'posQ_bg': bg_scale,
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


def parse_log(log_path):
    rows = []
    current = None
    waiting_class = None

    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for raw_line in f:
            line = raw_line.strip()

            combo_match = re.search(
                r'Combo\s+(\d+).*?FG Scale:\s*([0-9.eE+-]+).*?BG Scale:\s*([0-9.eE+-]+)',
                line)
            if combo_match:
                if current is not None:
                    rows.append(current)
                current = empty_row(
                    int(combo_match.group(1)),
                    float(combo_match.group(2)),
                    float(combo_match.group(3)))
                waiting_class = None
                continue

            if current is None:
                continue

            matched_header = False
            for cls_name, pattern in CLASS_HEADERS.items():
                if re.search(pattern, line):
                    if current.get(f'{cls_name}_3d_AP_R40_moderate') == '':
                        waiting_class = cls_name
                    else:
                        waiting_class = None
                    matched_header = True
                    break
            if matched_header:
                continue

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
    parser = argparse.ArgumentParser(description='Parse Split-GPCC AP values from test_split.py logs.')
    parser.add_argument('--log', required=True, help='Path to log_eval_split_*.txt')
    parser.add_argument('--out', required=True, help='Output AP CSV')
    args = parser.parse_args()

    rows = parse_log(args.log)
    if not rows:
        raise RuntimeError(f'No AP rows parsed from {args.log}')

    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f'Parsed {len(rows)} split AP rows -> {args.out}')


if __name__ == '__main__':
    main()
