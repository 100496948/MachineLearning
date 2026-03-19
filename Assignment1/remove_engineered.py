import csv
from pathlib import Path

INPUT = Path(__file__).parent / 'data_agent copy.csv'
OUTPUT = Path(__file__).parent / 'data_agent_copy_no_engineered.csv'

ENGINEERED = [
    'euclidean_distance',
    'total_velocity',
    'fast_descent',
    'polar_angle_to_pad',
    'abs_y_velocity',
    'angle_x_angular_vel',
    'x_pos_x_x_vel',
    'y_pos_x_y_vel',
    'low_alt_high_speed',
]


def main():
    with INPUT.open(newline='', encoding='utf-8') as fin:
        reader = csv.DictReader(fin)
        keep_cols = [c for c in reader.fieldnames if c not in ENGINEERED]

        with OUTPUT.open('w', newline='', encoding='utf-8') as fout:
            writer = csv.DictWriter(fout, fieldnames=keep_cols)
            writer.writeheader()
            rows_written = 0
            for r in reader:
                out = {k: r[k] for k in keep_cols}
                writer.writerow(out)
                rows_written += 1

    print(f'Input: {INPUT}')
    print(f'Output: {OUTPUT}')
    print(f'Columns removed: {len(ENGINEERED)}')
    print(f'Rows written: {rows_written}')


if __name__ == '__main__':
    main()
