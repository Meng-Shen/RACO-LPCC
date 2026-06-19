import argparse
import csv


def read_by_rate_id(path):
    with open(path, newline='') as f:
        return {int(row['rate_id']): row for row in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser(description='Merge baseline AP and G-PCC rate/time CSVs into curve point pairs.')
    parser.add_argument('--ap_csv', required=True)
    parser.add_argument('--gpcc_csv', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    ap_rows = read_by_rate_id(args.ap_csv)
    gpcc_rows = read_by_rate_id(args.gpcc_csv)
    rate_ids = sorted(set(ap_rows) & set(gpcc_rows))
    if not rate_ids:
        raise RuntimeError('No matching rate_id found between AP CSV and G-PCC CSV.')

    fieldnames = [
        'rate_id', 'scale', 'posQuantscale',
        'bpp', 'enc_time', 'dec_time',
        'Car_3d_AP_R40_moderate',
        'Pedestrian_3d_AP_R40_moderate',
        'Cyclist_3d_AP_R40_moderate',
    ]
    rows = []
    for rate_id in rate_ids:
        ap = ap_rows[rate_id]
        gpcc = gpcc_rows[rate_id]
        rows.append({
            'rate_id': rate_id,
            'scale': ap.get('scale') or gpcc.get('scale'),
            'posQuantscale': gpcc.get('posQuantscale') or ap.get('posQuantscale'),
            'bpp': gpcc.get('bpp'),
            'enc_time': gpcc.get('enc_time'),
            'dec_time': gpcc.get('dec_time'),
            'Car_3d_AP_R40_moderate': ap.get('Car_3d_AP_R40_moderate'),
            'Pedestrian_3d_AP_R40_moderate': ap.get('Pedestrian_3d_AP_R40_moderate'),
            'Cyclist_3d_AP_R40_moderate': ap.get('Cyclist_3d_AP_R40_moderate'),
        })

    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Merged {len(rows)} baseline curve rows -> {args.out}')


if __name__ == '__main__':
    main()
