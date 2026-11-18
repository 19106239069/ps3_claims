
import hashlib
import json


def create_sample_split(df, id_column, training_frac=0.8):
    """
    Create a reproducible train/test split based on ID columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    id_column : str or list
        Name of one ID column, or a list of ID columns.
    training_frac : float, optional
        Fraction of IDs assigned to the training set, by default 0.8.

    Returns
    -------
    pd.DataFrame
        A copy of the input data with a 'sample' column.
    """

    # Check the training fraction
    if not 0 <= training_frac <= 1:
        raise ValueError("training_frac must be between 0 and 1.")

    # Accept either one ID column or multiple ID columns
    if isinstance(id_column, str):
        id_columns = [id_column]
    else:
        id_columns = list(id_column)

    if not id_columns:
        raise ValueError("At least one ID column must be provided.")

    # Check that all requested columns exist
    missing_columns = [col for col in id_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Columns not found: {missing_columns}")

    # Missing IDs cannot be used for a reliable group split
    if df[id_columns].isna().any().any():
        raise ValueError("ID columns must not contain missing values.")

    def assign_sample(values):
        # Convert the ID values into a stable string representation
        key = json.dumps(
            [(type(value).__name__, str(value)) for value in values],
            ensure_ascii=False,
            separators=(",", ":"),
        )

        # Use SHA-256 to generate a reproducible hash
        hash_value = int(
            hashlib.sha256(key.encode("utf-8")).hexdigest(), 16
        )

        # Map the hash to [0, 1) and assign the sample
        fraction = hash_value / (1 << 256)

        return "train" if fraction < training_frac else "test"

    # Do not modify the original DataFrame
    result = df.copy()

    result["sample"] = [
        assign_sample(values)
        for values in result[id_columns].itertuples(index=False, name=None)
    ]

    return result


# Alternative function name used in the exercise description
create_sample_column = create_sample_split
