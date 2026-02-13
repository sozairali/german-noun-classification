"""
Data loading and cleaning for German noun classification.
"""

import re
import pandas as pd
from pathlib import Path
from typing import Tuple

from germannouns.config import NOUNS_CSV


def load_nouns(csv_path: Path = NOUNS_CSV) -> pd.DataFrame:
    """
    Load and clean German nouns dataset.

    Performs the following cleaning steps:
    1. Load CSV and select lemma and genus columns
    2. Drop rows with missing values
    3. Remove nouns starting with special characters or numbers
    4. Reset index

    Args:
        csv_path: Path to nouns.csv file (defaults to config.NOUNS_CSV)

    Returns:
        DataFrame with columns ['lemma', 'genus']
        - lemma: The German noun (e.g., 'Haus')
        - genus: Gender code ('f', 'm', or 'n')

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing

    Examples:
        >>> nouns = load_nouns()
        >>> 'lemma' in nouns.columns and 'genus' in nouns.columns
        True
        >>> nouns['genus'].isin(['f', 'm', 'n']).all()
        True
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Nouns CSV not found at: {csv_path}")

    # Load CSV
    nouns = pd.read_csv(csv_path)

    # Validate required columns
    if 'lemma' not in nouns.columns or 'genus' not in nouns.columns:
        raise ValueError("CSV must contain 'lemma' and 'genus' columns")

    # Select only required columns
    nouns = nouns[["lemma", "genus"]]

    # Drop missing values
    nouns = nouns.dropna()

    # Clean data: remove nouns starting with special characters or numbers
    # Vectorized version (more efficient than loop)
    nouns = nouns[~nouns["lemma"].str.match(r"^[\W\d]", na=False)]

    # Drop any remaining missing values and reset index
    nouns = nouns.dropna().reset_index(drop=True)

    return nouns


def get_gender_distribution(nouns: pd.DataFrame) -> pd.Series:
    """
    Calculate gender distribution proportions from nouns dataset.

    Args:
        nouns: DataFrame with 'genus' column

    Returns:
        Series with proportions for each gender ('f', 'm', 'n')

    Examples:
        >>> nouns = pd.DataFrame({'genus': ['f', 'f', 'm', 'n']})
        >>> dist = get_gender_distribution(nouns)
        >>> dist['f']
        0.5
    """
    return nouns['genus'].value_counts(normalize=True)
