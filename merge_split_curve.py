import argparse
import csv


def scale_key(row):
    if row.get('posQ_fg') not in (None, '') and row.get('posQ_bg') not in (None, ''):
        return (round(float(row['posQ_fg']), 6), round(float(row['posQ_bg']), 6))

    scale = row.get('scale') or row.get('posQuantscale')
    if scale:
        parts = [part.strip() for part in scale.split(',')]
        if len(parts) == 2:
            return (round(float(parts[0]), 6), round(float(parts[1]), 6))

    raise ValueError(f'Cannot get split scale key from row: {row}')


def read_rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def read_by_scale(path):
    rows = read_rows(path)
    return {scale_key(row): row for row in rows}


def main():
    parser = argparse.ArgumentParser(description='Merge Split-GPCC AP and rate/time CSVs into curve point pairs.')
    parser.add_argument('--ap_csv', required=True)
    parser.add_argument('--gpcc_csv', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    ap_rows = read_rows(args.ap_csv)
    gpcc_rows = read_by_scale(args.gpcc_csv)
    if not ap_rows or not gpcc_rows:
        raise RuntimeError('No AP or Split-GPCC rows found.')

    fieldnames = [
        'rate_id', 'combo_id', 'scale', 'posQuantscale', 'posQ_fg', 'posQ_bg',
        'bpp', 'seg_time', 'fg_enc_time', 'bg_enc_time', 'gpcc_enc_time',
        'enc_time', 'dec_time',
        'Car_3d_AP_R40_moderate',
        'Pedestrian_3d_AP_R40_moderate',
        'Cyclist_3d_AP_R40_moderate',
    ]
    rows = []
    for ap in ap_rows:
        key = scale_key(ap)
        if key not in gpcc_rows:
            continue
        gpcc = gpcc_rows[key]
        rate_id = ap.get('rate_id') or gpcc.get('rate_id')
        rows.append({
            'rate_id': rate_id,
            'combo_id': gpcc.get('combo_id') or ap.get('combo_id') or rate_id,
            'scale': gpcc.get('scale') or ap.get('scale'),
            'posQuantscale': gpcc.get('posQuantscale') or ap.get('posQuantscale'),
            'posQ_fg': gpcc.get('posQ_fg') or ap.get('posQ_fg'),
            'posQ_bg': gpcc.get('posQ_bg') or ap.get('posQ_bg'),
            'bpp': gpcc.get('bpp'),
            'seg_time': gpcc.get('seg_time'),
            'fg_enc_time': gpcc.get('fg_enc_time'),
            'bg_enc_time': gpcc.get('bg_enc_time'),
            'gpcc_enc_time': gpcc.get('gpcc_enc_time'),
            'enc_time': gpcc.get('enc_time'),
            'dec_time': gpcc.get('dec_time'),
            'Car_3d_AP_R40_moderate': ap.get('Car_3d_AP_R40_moderate'),
            'Pedestrian_3d_AP_R40_moderate': ap.get('Pedestrian_3d_AP_R40_moderate'),
            'Cyclist_3d_AP_R40_moderate': ap.get('Cyclist_3d_AP_R40_moderate'),
        })

    if not rows:
        raise RuntimeError('No matching split scale found between AP and Split-GPCC CSVs.')

    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Merged {len(rows)} Split-GPCC curve rows -> {args.out}')


if __name__ == '__main__':
    main()
