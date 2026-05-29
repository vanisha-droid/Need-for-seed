import pandas as pd
import numpy as np


def csv_to_binary_dataset(
    input_csv,
    output_csv=None,
    n=200,
    d=20,
    label_columns=None,
    threshold=127,
    random_state=42,
):
    """
    Convert a CSV dataset into a binary NumPy-style dataset with shape (n, d),
    matching the format returned by:

        generate_sample_data(n=200, d=20, p=0.3)

    Final output:
        - exactly n rows
        - exactly d features
        - values only 0 or 1

    Parameters
    ----------
    input_csv : str
        Path to input CSV file.

    output_csv : str or None
        Optional path to save processed dataset.

    n : int
        Number of rows.

    d : int
        Number of features.

    label_columns : list or None
        Columns to drop (e.g. labels such as 'label').

    threshold : int or float
        Threshold used to binarise values.
        Values > threshold become 1, otherwise 0.

    random_state : int
        Random seed.

    Returns
    -------
    np.ndarray
        Binary dataset of shape (n, d).
    """

    # -----------------------------
    # Load CSV
    # -----------------------------
    df = pd.read_csv(input_csv)

    # -----------------------------
    # Remove label columns if needed
    # -----------------------------
    if label_columns is not None:
        df = df.drop(columns=label_columns, errors="ignore")

    # -----------------------------
    # Keep only numeric columns
    # -----------------------------
    df = df.select_dtypes(include=[np.number])

    # -----------------------------
    # Ensure enough rows
    # -----------------------------
    if len(df) < n:
        raise ValueError(f"Dataset only has {len(df)} rows, but n={n} requested.")

    # -----------------------------
    # Randomly sample n rows
    # -----------------------------
    df = df.sample(n=n, random_state=random_state).reset_index(drop=True)

    # -----------------------------
    # Ensure exactly d columns
    # -----------------------------
    if df.shape[1] < d:
        raise ValueError(
            f"Dataset only has {df.shape[1]} numeric columns, but d={d} requested."
        )

    # Use first d features
    df = df.iloc[:, :d]

    # -----------------------------
    # Convert to binary (0/1)
    # -----------------------------
    binary_df = (df > threshold).astype(int)

    # -----------------------------
    # Convert to NumPy array
    # -----------------------------
    data = binary_df.to_numpy(dtype=int)

    # -----------------------------
    # Save if requested
    # -----------------------------
    if output_csv is not None:
        pd.DataFrame(data).to_csv(output_csv, index=False)
        print(f"Saved processed dataset to: {output_csv}")

    return data


# -------------------------------------------------
# Example usage for MNIST 0/1 dataset
# -------------------------------------------------
def x():

    # load raw .data file
    df = pd.read_csv(
        "agaricus-lepiota.data",
        header=None
    )

    print(df.head())

    binary_df = pd.get_dummies(df)

    data = (
    binary_df
    .sample(n=200, random_state=42)
    .iloc[:, :20]
    .to_numpy(dtype=int)
    )

    return data


    # Matches generate_sample_data output format:
    # array([[0,1,1,...],
    #        [1,0,0,...],
    #        ...])
