"""
VQQL Discretization
January 2026 - Machine Learning Classes
University Carlos III of Madrid

Modified version that loads TWO CSV files (keyboard + agent) and mixes them
into a single training dataset for the vector quantizer, as suggested by
the original assignment design.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import joblib
import os

# =========================================================
# PHASE 1: State Representation with Vector Quantization
# =========================================================

# ==========================================
# CONFIGURATION PARAMETERS
# ==========================================

# Input CSV files
KEYBOARD_CSV = "keyboard_data.csv"
AGENT_CSV    = "agent_data.csv"

# Output file where the vector quantizer will be stored
OUTPUT_FILE = "lunarlander_vq.pkl"

# Columns that represent the STATE of the environment
# IMPORTANT: Do NOT include the action column here. It must match exactly
# the order used in extract_features() in LunarLander_RL.py.
STATE_COLUMNS = [
    "x_position",
    "y_position",
    "x_velocity",
    "y_velocity",
    "angle",
    "angular_velocity",
    "left_leg",
    "right_leg"
]

# Number of clusters (i.e., number of discrete states).
# Try 32, 64, 128, 256 to study the effect of granularity.
N_CLUSTERS = 256

# Number of samples to use from EACH dataset (keyboard and agent)
SAMPLES_PER_SOURCE = 25000


# ==========================================
# LOAD AND PREPARE DATA
# ==========================================

def load_and_mix_states(keyboard_csv, agent_csv, state_columns, samples_per_source):
    """
    Load state data from one or two CSV files (keyboard and agent),
    select relevant columns, and mix them into a single dataset.

    If one of the files is missing, the other is used alone (with a warning).
    """
    frames = []

    if os.path.exists(keyboard_csv):
        df_kb = pd.read_csv(keyboard_csv)[state_columns].dropna()
        n_kb = min(samples_per_source, len(df_kb))
        df_kb = df_kb.sample(n=n_kb, random_state=42)
        print(f"  Keyboard data: {len(df_kb)} rows (from {keyboard_csv})")
        frames.append(df_kb)
    else:
        print(f"  WARNING: '{keyboard_csv}' not found. Skipping keyboard data.")

    if os.path.exists(agent_csv):
        df_ag = pd.read_csv(agent_csv)[state_columns].dropna()
        n_ag = min(samples_per_source, len(df_ag))
        df_ag = df_ag.sample(n=n_ag, random_state=42)
        print(f"  Agent data:    {len(df_ag)} rows (from {agent_csv})")
        frames.append(df_ag)
    else:
        print(f"  WARNING: '{agent_csv}' not found. Skipping agent data.")

    if not frames:
        raise FileNotFoundError(
            "No CSV input file found. At least one of "
            f"'{keyboard_csv}' or '{agent_csv}' must exist."
        )

    # Concatenate and shuffle so data is well mixed
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"  Total mixed dataset: {len(df)} rows")
    return df.to_numpy(dtype=np.float64)


# ==========================================
# BUILD VECTOR QUANTIZER
# ==========================================

def build_quantizer(states, n_clusters):
    """
    Train a vector quantizer (MiniBatchKMeans) on the state data.
    """
    if len(states) < n_clusters:
        raise ValueError("Number of samples must be >= number of clusters.")

    # Step 1: scale the data so all features have comparable influence
    scaler = StandardScaler()
    states_scaled = scaler.fit_transform(states)

    # Step 2: train MiniBatchKMeans
    quantizer = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=1024,
        n_init=10,
    )
    quantizer.fit(states_scaled)

    return scaler, quantizer


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("=" * 60)
    print("VQQL Discretization - training the vector quantizer")
    print("=" * 60)
    print(f"Number of clusters: {N_CLUSTERS}")
    print(f"Samples per source: {SAMPLES_PER_SOURCE}")
    print()
    print("Loading and mixing state data...")

    states = load_and_mix_states(
        KEYBOARD_CSV,
        AGENT_CSV,
        STATE_COLUMNS,
        SAMPLES_PER_SOURCE,
    )

    print(f"\nStates shape: {states.shape}")

    print("\nTraining vector quantizer...")
    scaler, quantizer = build_quantizer(states, N_CLUSTERS)

    # Show centroids in original (unscaled) space for inspection
    centroids = scaler.inverse_transform(quantizer.cluster_centers_)
    df_centroids = pd.DataFrame(centroids, columns=STATE_COLUMNS)
    print("\nFirst 10 centroids (in original units):")
    print(df_centroids.head(10).to_string(index=False))

    print("\nSaving quantizer model...")
    joblib.dump(
        {
            "scaler": scaler,
            "quantizer": quantizer,
            "state_columns": STATE_COLUMNS,
            "n_clusters": N_CLUSTERS,
        },
        OUTPUT_FILE,
    )

    print(f"Quantizer saved to {OUTPUT_FILE}")
    print("Phase 2 quantizer training completed successfully!")


if __name__ == "__main__":
    main()