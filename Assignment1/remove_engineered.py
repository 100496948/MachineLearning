import csv
from pathlib import Path
import argparse
import sys
import re
import io
import csv as _csv


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
    'next_reward',
]


def process_file(path: Path) -> int:
    if not path.exists():
        print(f'File not found: {path}', file=sys.stderr)
        return 0
    out_path = path.with_name(path.stem + '_no_engineered' + path.suffix)

    if path.suffix.lower() == '.csv':
        rows_written = 0
        with path.open(newline='', encoding='utf-8') as fin:
            reader = _csv.DictReader(fin)
            if reader.fieldnames is None:
                print(f'No header found in {path}', file=sys.stderr)
                return 0

            keep_cols = [c for c in reader.fieldnames if c not in ENGINEERED]

            with out_path.open('w', newline='', encoding='utf-8') as fout:
                writer = _csv.DictWriter(fout, fieldnames=keep_cols)
                writer.writeheader()
                for r in reader:
                    out = {k: r.get(k, '') for k in keep_cols}
                    writer.writerow(out)
                    rows_written += 1

        print(f'Processed CSV: {path} -> {out_path}; removed: {len(reader.fieldnames) - len(keep_cols)}; rows: {rows_written}')
        return rows_written

    if path.suffix.lower() == '.arff':
        # parse ARFF header and attributes
        with path.open('r', encoding='utf-8') as fin:
            lines = fin.readlines()

        header_lines = []
        attr_defs = []  # tuples (name, def_line)
        data_lines = []
        in_data = False

        attr_re = re.compile(r"@attribute\s+('([^']+)'|\"([^\"]+)\"|([^\s]+))\s+(.*)", re.IGNORECASE)

        for ln in lines:
            l = ln.rstrip('\n')
            if in_data:
                if l.strip() == '' or l.strip().startswith('%'):
                    continue
                data_lines.append(l)
            else:
                header_lines.append(l)
                if l.strip().lower().startswith('@data'):
                    in_data = True
                    continue
                m = attr_re.match(l.strip())
                if m:
                    name = m.group(2) or m.group(3) or m.group(4)
                    attr_defs.append((name, l))

        if not attr_defs:
            print(f'No @attribute lines found in {path}', file=sys.stderr)
            return 0

        keep_indices = [i for i, (n, _) in enumerate(attr_defs) if n not in ENGINEERED]

        # write new ARFF
        rows_written = 0
        with out_path.open('w', encoding='utf-8', newline='') as fout:
            # write preamble until first @attribute
            wrote_attrs = False
            for ln in header_lines:
                if not wrote_attrs and ln.strip().lower().startswith('@attribute'):
                    # write only kept attribute lines
                    for idx in keep_indices:
                        fout.write(attr_defs[idx][1] + '\n')
                    wrote_attrs = True
                    continue
                # skip original attribute lines
                if ln.strip().lower().startswith('@attribute'):
                    continue
                fout.write(ln + '\n')

            # ensure @data line
            if not any(l.strip().lower().startswith('@data') for l in header_lines):
                fout.write('@data\n')

            # write data rows, parsing with csv to handle commas
            reader = _csv.reader(data_lines)
            writer = _csv.writer(fout)
            for row in reader:
                if len(row) < len(attr_defs):
                    # pad if needed
                    row += [''] * (len(attr_defs) - len(row))
                newrow = [row[i].strip() for i in keep_indices]
                writer.writerow(newrow)
                rows_written += 1

        print(f'Processed ARFF: {path} -> {out_path}; removed attributes: {len(attr_defs) - len(keep_indices)}; rows: {rows_written}')
        return rows_written

    print(f'Unsupported file type: {path}', file=sys.stderr)
    return 0


def main():
    parser = argparse.ArgumentParser(description='Remove engineered feature columns from CSV and ARFF files')
    parser.add_argument('files', nargs='*', help='CSV files to process (default: all .csv in script dir)')
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    if not args.files:
        files = list(base_dir.glob('*.csv')) + list(base_dir.glob('*.arff'))
    else:
        files = [Path(f) for f in args.files]

    total = 0
    for f in files:
        total += process_file(f)

    print(f'Total rows written across files: {total}')


if __name__ == '__main__':
    main()
