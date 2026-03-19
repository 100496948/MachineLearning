import re
from pathlib import Path

INPUT = Path(__file__).parent / 'data_agent copy.arff'
OUTPUT = Path(__file__).parent / 'data_agent_copy_no_engineered.arff'

ENGINEERED = {
    'euclidean_distance',
    'total_velocity',
    'fast_descent',
    'polar_angle_to_pad',
    'abs_y_velocity',
    'angle_x_angular_vel',
    'x_pos_x_x_vel',
    'y_pos_x_y_vel',
    'low_alt_high_speed',
}


def parse_arff(path: Path):
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines()

    header = []
    attributes = []
    data_lines = []
    in_data = False

    attr_re = re.compile(r"^@attribute\s+('.*?'|\".*?\"|\S+)\s+.*", re.IGNORECASE)

    for ln in lines:
        if not in_data:
            header.append(ln)
            if ln.strip().lower().startswith('@attribute'):
                m = attr_re.match(ln.strip())
                if m:
                    name = m.group(1)
                    if (name.startswith("'") and name.endswith("'")) or (name.startswith('"') and name.endswith('"')):
                        name = name[1:-1]
                    attributes.append(name)
            if ln.strip().lower().startswith('@data'):
                in_data = True
        else:
            if ln.strip() == '':
                continue
            if ln.strip().startswith('%'):
                continue
            data_lines.append(ln)

    return header, attributes, data_lines


def write_arff(output: Path, header_lines, attributes, keep_idx, data_lines):
    with output.open('w', encoding='utf-8', newline='') as fout:
        # write relation and non-attribute header, but rebuild attribute section with kept attributes
        written_attrs = 0
        for ln in header_lines:
            if ln.strip().lower().startswith('@attribute'):
                # only write attributes we kept (in order)
                if written_attrs in keep_idx:
                    fout.write(ln + '\n')
                written_attrs += 1
            else:
                fout.write(ln + '\n')

        fout.write('\n')
        fout.write('@DATA\n')
        for ln in data_lines:
            parts = [p.strip() for p in ln.split(',')]
            out_parts = [parts[i] for i in keep_idx]
            fout.write(','.join(out_parts) + '\n')


def main():
    if not INPUT.exists():
        print('Input ARFF not found:', INPUT)
        return

    header, attributes, data_lines = parse_arff(INPUT)
    attr_lower = [a.lower() for a in attributes]
    keep_idx = [i for i, a in enumerate(attr_lower) if a not in ENGINEERED]

    removed = [attributes[i] for i in range(len(attributes)) if i not in keep_idx]

    write_arff(OUTPUT, header, attributes, keep_idx, data_lines)

    print(f'Input: {INPUT}')
    print(f'Output: {OUTPUT}')
    print(f'Attributes total: {len(attributes)}')
    print(f'Removed attributes: {len(removed)} -> {removed}')
    print(f'Rows: {len(data_lines)}')


if __name__ == '__main__':
    main()
