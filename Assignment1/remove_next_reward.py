from pathlib import Path
import csv
import sys

ROOT = Path(__file__).parent

TARGETS = [
    ROOT / 'data_agent.csv',
    ROOT / 'data_agent.arff',
    ROOT / 'data_keyboard.csv',
    ROOT / 'data_keyboard.arff',
]


def process_csv(path: Path):
    out_path = path.with_name(path.stem + '_no_next_reward' + path.suffix)
    with path.open(newline='', encoding='utf-8') as fin:
        reader = csv.DictReader(fin)
        if 'next_reward' not in reader.fieldnames:
            print(f'skipping {path.name}: no next_reward column')
            return
        keep = [c for c in reader.fieldnames if c != 'next_reward']
        with out_path.open('w', newline='', encoding='utf-8') as fout:
            writer = csv.DictWriter(fout, fieldnames=keep)
            writer.writeheader()
            for r in reader:
                writer.writerow({k: r[k] for k in keep})
    print(f'Wrote {out_path.name}')


def process_arff(path: Path):
    out_path = path.with_name(path.stem + '_no_next_reward' + path.suffix)
    lines = path.read_text(encoding='utf-8').splitlines()
    header = []
    data = []
    in_data = False
    removed_attribute = False
    for ln in lines:
        if not in_data and ln.strip().lower().startswith('@data'):
            in_data = True
            header.append(ln)
            continue
        if not in_data:
            # attribute lines
            if ln.strip().lower().startswith('@attribute') and 'next_reward' in ln.lower():
                removed_attribute = True
                continue
            header.append(ln)
        else:
            if not ln.strip() or ln.strip().startswith('%'):
                data.append(ln)
                continue
            # remove last comma-separated value (assumes next_reward is last)
            if ',' in ln:
                new_ln = ','.join(ln.split(',')[:-1])
                data.append(new_ln)
            else:
                data.append(ln)

    if not removed_attribute:
        print(f'skipping {path.name}: no next_reward attribute found')
        return

    out_text = '\n'.join(header + data) + '\n'
    out_path.write_text(out_text, encoding='utf-8')
    print(f'Wrote {out_path.name}')


def main(files=None):
    targets = files or TARGETS
    for f in targets:
        p = Path(f)
        if not p.exists():
            print(f'missing {p.name}, skipping')
            continue
        if p.suffix.lower() == '.csv':
            process_csv(p)
        elif p.suffix.lower() == '.arff':
            process_arff(p)
        else:
            print(f'unsupported {p.name}, skipping')


if __name__ == '__main__':
    args = sys.argv[1:]
    if args:
        main(args)
    else:
        main()
