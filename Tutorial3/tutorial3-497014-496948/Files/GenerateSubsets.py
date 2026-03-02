"""
Script to generate filtered ARFF files from all_data_lunarlander.arff

Creates two files with different subsets of attributes:
- filter_data_lunarlander_manual1.arff: position and velocity attributes
- filter_data_lunarlander_manual2.arff: angle and leg contact attributes

Both files keep the 'action' class attribute.
"""

INPUT_FILE = "all_data_lunarlander.arff"

# Define which column indices to keep for each filter
# Original columns: 0:x_pos, 1:y_pos, 2:x_vel, 3:y_vel, 4:angle, 5:angular_vel, 6:left_leg, 7:right_leg, 8:action

# Filter 1: position and velocity
FILTER1_COLUMNS = [0, 1, 2, 3, 8]  # x_pos, y_pos, x_vel, y_vel, action
FILTER1_FILE = "filter_data_lunarlander_manual1.arff"
FILTER1_HEADER = (
    "@RELATION lunar_lander_filter1\n"
    "\n"
    "@ATTRIBUTE x_position    NUMERIC\n"
    "@ATTRIBUTE y_position    NUMERIC\n"
    "@ATTRIBUTE x_velocity    NUMERIC\n"
    "@ATTRIBUTE y_velocity    NUMERIC\n"
    "@ATTRIBUTE action        {nothing, left_engine, main_engine, right_engine}\n"
    "\n"
    "@DATA\n"
)

# Filter 2: angle and leg contacts
FILTER2_COLUMNS = [4, 5, 6, 7, 8]  # angle, angular_vel, left_leg, right_leg, action
FILTER2_FILE = "filter_data_lunarlander_manual2.arff"
FILTER2_HEADER = (
    "@RELATION lunar_lander_filter2\n"
    "\n"
    "@ATTRIBUTE angle         NUMERIC\n"
    "@ATTRIBUTE angular_vel   NUMERIC\n"
    "@ATTRIBUTE left_leg      {0, 1}\n"
    "@ATTRIBUTE right_leg     {0, 1}\n"
    "@ATTRIBUTE action        {nothing, left_engine, main_engine, right_engine}\n"
    "\n"
    "@DATA\n"
)


def read_data_lines(filepath):
    """Read only the data lines from an ARFF file (everything after @DATA)."""
    data_lines = []
    in_data = False
    with open(filepath, "r") as f:
        for line in f:
            if in_data:
                stripped = line.strip()
                if stripped and not stripped.startswith("%"):
                    data_lines.append(stripped)
            elif line.strip().upper() == "@DATA":
                in_data = True
    return data_lines


def filter_columns(data_lines, columns):
    """Keep only the specified column indices from each data line."""
    filtered = []
    for line in data_lines:
        values = line.split(",")
        selected = [values[i] for i in columns]
        filtered.append(",".join(selected))
    return filtered


def write_arff(filepath, header, data_lines):
    """Write a complete ARFF file with header and data."""
    with open(filepath, "w") as f:
        f.write(header)
        for line in data_lines:
            f.write(line + "\n")


def main():
    print(f"Reading data from {INPUT_FILE}...")
    data_lines = read_data_lines(INPUT_FILE)
    print(f"Found {len(data_lines)} instances.\n")

    # Generate filter 1
    filtered1 = filter_columns(data_lines, FILTER1_COLUMNS)
    write_arff(FILTER1_FILE, FILTER1_HEADER, filtered1)
    print(f"Created {FILTER1_FILE} with {len(filtered1)} instances (position + velocity).")

    # Generate filter 2
    filtered2 = filter_columns(data_lines, FILTER2_COLUMNS)
    write_arff(FILTER2_FILE, FILTER2_HEADER, filtered2)
    print(f"Created {FILTER2_FILE} with {len(filtered2)} instances (angle + leg contacts).")

    print("\nDone!")


if __name__ == "__main__":
    main()