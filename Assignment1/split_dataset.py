"""
split_dataset.py
Splits the generated LunarLander datasets into 80% train / 20% test.

Usage:
    python split_dataset.py data_agent
    python split_dataset.py data_keyboard

This will read data_agent.arff + data_agent.csv and produce:
    training_agent.arff / test_agent.arff
    training_agent.csv  / test_agent.csv
"""

import sys
import random
import os

# Reproducible split
RANDOM_SEED = 42
TRAIN_RATIO = 0.8


def read_arff(filepath):
    """
    Read an ARFF file and separate the header from the data lines.
    Returns (header_lines, data_lines) where both are lists of strings.
    """
    header_lines = []
    data_lines = []
    in_data = False

    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()
            if not in_data:
                header_lines.append(line)
                if stripped.upper() == "@DATA":
                    in_data = True
            else:
                if stripped:  # skip empty lines
                    data_lines.append(line)

    return header_lines, data_lines


def read_csv(filepath):
    """
    Read a CSV file and separate header from data lines.
    Returns (header_line, data_lines).
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    header_line = lines[0]
    data_lines = [l for l in lines[1:] if l.strip()]

    return header_line, data_lines


def write_lines(filepath, header, data_lines):
    """Write header + data lines to a file."""
    with open(filepath, "w") as f:
        if isinstance(header, list):
            # ARFF header is multiple lines
            for h in header:
                f.write(h)
        else:
            # CSV header is a single line
            f.write(header)
        for line in data_lines:
            f.write(line)


def split_dataset(base_name):
    """
    Split base_name.arff and base_name.csv into train/test files.

    Input:   data_agent.arff,  data_agent.csv
                 or
             data_keyboard.arff, data_keyboard.csv

    Output:  training_agent.arff, test_agent.arff,
             training_agent.csv,  test_agent.csv
                 (or _keyboard variants)
    """
    # Determine the suffix: "agent" or "keyboard"
    if "agent" in base_name:
        suffix = "agent"
    elif "keyboard" in base_name:
        suffix = "keyboard"
    else:
        suffix = base_name

    arff_path = f"{base_name}.arff"
    csv_path = f"{base_name}.csv"

    # Check files exist
    if not os.path.exists(arff_path):
        print(f"Error: {arff_path} not found.")
        return
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Read files
    arff_header, arff_data = read_arff(arff_path)
    csv_header, csv_data = read_csv(csv_path)

    # Verify same number of instances
    assert len(arff_data) == len(csv_data), (
        f"Mismatch: ARFF has {len(arff_data)} instances, "
        f"CSV has {len(csv_data)} instances."
    )

    n = len(arff_data)
    print(f"Total instances: {n}")

    # Generate shuffled indices
    random.seed(RANDOM_SEED)
    indices = list(range(n))
    random.shuffle(indices)

    # Split
    split_point = int(n * TRAIN_RATIO)
    train_idx = indices[:split_point]
    test_idx = indices[split_point:]

    print(f"Training instances: {len(train_idx)} ({TRAIN_RATIO*100:.0f}%)")
    print(f"Test instances:     {len(test_idx)} ({(1-TRAIN_RATIO)*100:.0f}%)")

    # Select lines by index
    arff_train = [arff_data[i] for i in train_idx]
    arff_test = [arff_data[i] for i in test_idx]
    csv_train = [csv_data[i] for i in train_idx]
    csv_test = [csv_data[i] for i in test_idx]

    # Write output files
    write_lines(f"training_{suffix}.arff", arff_header, arff_train)
    write_lines(f"test_{suffix}.arff", arff_header, arff_test)
    write_lines(f"training_{suffix}.csv", csv_header, csv_train)
    write_lines(f"test_{suffix}.csv", csv_header, csv_test)

    print(f"\nFiles created:")
    print(f"  training_{suffix}.arff  /  training_{suffix}.csv")
    print(f"  test_{suffix}.arff      /  test_{suffix}.csv")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_dataset.py <base_name>")
        print("  e.g: python split_dataset.py data_agent")
        print("       python split_dataset.py data_keyboard")
        sys.exit(1)

    split_dataset(sys.argv[1])