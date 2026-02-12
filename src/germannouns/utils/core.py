"""
Core utility functions for German noun classification.

Provides n-gram generation, statistical testing, and data transformation utilities.
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

from germannouns.config import START_TOKEN, END_TOKEN, MIN_NGRAM_COUNT


def create_ngram(word: str, n: int) -> List[str]:
    """
    Create n-grams from a word with start and end tokens.

    For n=1, returns individual characters plus start/end tokens.
    For n>1, returns sliding window n-grams as concatenated strings.

    Args:
        word: The word to generate n-grams from
        n: The size of n-grams to generate (1 for unigrams, 2 for bigrams, etc.)

    Returns:
        List of n-gram strings. For n=1: ['<S>', 'w', 'o', 'r', 'd', '<E>']
        For n=2: ['<Sw', 'wo', 'or', 'rd', 'd<E>'], etc.

    Examples:
        >>> create_ngram("ab", 1)
        ['<S>', 'a', 'b', '<E>']
        >>> create_ngram("ab", 2)
        ['<Sa', 'ab', 'b<E>']
        >>> create_ngram("ab", 3)
        ['<Sab', 'ab<E>']
    """
    # Add start and end tokens
    chr = [START_TOKEN] + list(str(word)) + [END_TOKEN]

    # For unigrams, return the character list directly
    if n == 1:
        return chr

    # For n-grams, create sliding windows
    chs = []
    for i in range(0, n):
        chs.append(chr[i:])

    # Zip together to create n-grams and convert tuples to strings
    chs = list(tuple(zip(*chs)))
    string = [''.join(ch) for ch in chs]

    return string


def calculate_p_value(row: pd.Series, expected_proportions: pd.Series) -> float:
    """
    Calculate chi-square test p-value for n-gram gender distribution.

    Tests whether the observed gender distribution for an n-gram differs
    significantly from the expected distribution across all nouns.

    Args:
        row: DataFrame row containing 'f', 'm', 'n' counts for an n-gram
        expected_proportions: Series with expected proportions for each gender

    Returns:
        P-value from chi-square test, or np.nan if expected counts are too low
        (minimum expected count must be > MIN_NGRAM_COUNT for valid test)

    Examples:
        >>> expected_props = pd.Series({'f': 0.5, 'm': 0.3, 'n': 0.2})
        >>> row = pd.Series({'f': 100, 'm': 60, 'n': 40, 'n-gram': 'ab'})
        >>> p_val = calculate_p_value(row, expected_props)
        >>> 0.0 <= p_val <= 1.0
        True
    """
    # Convert to numeric arrays to avoid dtype issues
    observed = row[['f', 'm', 'n']].astype(float).values
    total = row['f'] + row['m'] + row['n']
    expected = (expected_proportions * total).astype(float).values

    # Check if all expected counts are above minimum threshold
    if all(count > MIN_NGRAM_COUNT for count in expected):
        _, p_value, _, _ = chi2_contingency([observed, expected])
        return p_value
    else:
        return np.nan


def create_dictsof_counts(df: pd.DataFrame, col: pd.Series, targetcol: str) -> Dict[str, Dict[str, int]]:
    """
    Create nested dictionary of n-gram counts by category.

    For each n-gram, counts how many times it appears in each category
    (e.g., how often 'er' appears in feminine, masculine, neuter nouns).

    Args:
        df: DataFrame containing the target column
        col: Series where each cell contains a list of n-grams
        targetcol: Name of column containing categories (e.g., 'genus')

    Returns:
        Nested dict: {n-gram: {category: count}}
        Example: {'<Se': {'f': 150, 'm': 200, 'n': 100}, ...}

    Examples:
        >>> df = pd.DataFrame({'genus': ['f', 'm'], 'ngrams': [['ab', 'bc'], ['ab', 'cd']]})
        >>> counts = create_dictsof_counts(df, df['ngrams'], 'genus')
        >>> counts['ab']
        {'f': 1, 'm': 1}
    """
    absolute_values = {}

    for idx, cell in enumerate(col):
        category = df.iloc[idx][targetcol]
        for ngram in cell:
            # Initialize n-gram dict if not exists
            if ngram not in absolute_values:
                absolute_values[ngram] = {}
            # Increment count for this n-gram + category combination
            absolute_values[ngram][category] = absolute_values[ngram].get(category, 0) + 1

    return absolute_values


def convert_dicts_to_df(list_of_dicts: List[Dict[str, Dict[str, int]]]) -> pd.DataFrame:
    """
    Convert list of nested count dictionaries to a single DataFrame.

    Takes multiple count dictionaries (one per n-gram size) and combines them
    into a single DataFrame where each row is an n-gram and columns are categories.

    Args:
        list_of_dicts: List of nested dicts from create_dictsof_counts

    Returns:
        DataFrame with n-grams as index and categories as columns
        Columns: 'f', 'm', 'n' (counts for each gender)
        Index: n-gram strings

    Examples:
        >>> dict1 = {'ab': {'f': 10, 'm': 5}}
        >>> dict2 = {'abc': {'f': 3, 'm': 2, 'n': 1}}
        >>> df = convert_dicts_to_df([dict1, dict2])
        >>> 'ab' in df.index and 'abc' in df.index
        True
    """
    df = pd.DataFrame()

    for count_dict in list_of_dicts:
        # Convert nested dict to DataFrame with n-grams as index
        temp_df = pd.DataFrame.from_dict(count_dict, orient='index')
        df = pd.concat([df, temp_df], axis=0)

    return df
