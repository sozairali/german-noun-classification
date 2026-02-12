"""
Feature engineering for German noun classification.

Generates n-gram features, performs chi-square feature selection,
and creates one-hot encoded feature matrix for model training.
"""

from typing import Tuple, List
import pandas as pd

from germannouns.config import (
    NGRAM_RANGE,
    NGRAM_COLUMN_NAMES,
    P_VALUE_THRESHOLD,
    START_TOKEN,
    END_TOKEN,
    FILTER_END_TOKEN_ONLY
)
from germannouns.utils.core import (
    create_ngram,
    create_dictsof_counts,
    convert_dicts_to_df,
    calculate_p_value
)


def generate_ngrams(nouns: pd.DataFrame) -> pd.DataFrame:
    """
    Generate n-grams for all nouns in dataset.

    Creates n-gram columns (Chs, Chs2, Chs3, etc.) where each cell
    contains a list of n-grams for that word.

    Args:
        nouns: DataFrame with 'lemma' column

    Returns:
        DataFrame with original columns plus n-gram columns
        (Chs, Chs2, Chs3, etc.)

    Examples:
        >>> nouns = pd.DataFrame({'lemma': ['Haus'], 'genus': ['n']})
        >>> nouns_with_ngrams = generate_ngrams(nouns)
        >>> 'Chs' in nouns_with_ngrams.columns
        True
    """
    nouns = nouns.copy()

    # Generate n-grams for each n in range
    for n in range(1, NGRAM_RANGE[1]):
        col_name = NGRAM_COLUMN_NAMES[n]
        nouns[col_name] = nouns['lemma'].apply(lambda word: create_ngram(word, n))

    return nouns


def calculate_ngram_counts(nouns_with_ngrams: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate absolute counts of n-grams by gender.

    For each n-gram, counts how many times it appears in each gender category.

    Args:
        nouns_with_ngrams: DataFrame with n-gram columns from generate_ngrams()

    Returns:
        DataFrame with n-grams as index and gender counts as columns
        Columns: 'f', 'm', 'n', plus index column 'n-gram'

    Examples:
        >>> nouns = pd.DataFrame({'lemma': ['ab'], 'genus': ['f'],
        ...                       'Chs': [['<S>', 'a', 'b', '<E>']]})
        >>> counts = calculate_ngram_counts(nouns)
        >>> 'f' in counts.columns and 'm' in counts.columns
        True
    """
    # Create list of count dictionaries for each n-gram size
    lst_absolute_counts = []

    for n in range(1, NGRAM_RANGE[1]):
        col_name = NGRAM_COLUMN_NAMES[n]
        absolute_values = create_dictsof_counts(
            nouns_with_ngrams,
            nouns_with_ngrams[col_name],
            'genus'
        )
        lst_absolute_counts.append(absolute_values.copy())

    # Convert to DataFrame
    absolute_counts_df = convert_dicts_to_df(lst_absolute_counts)

    # Reset index and rename
    absolute_counts_df.reset_index(inplace=True)
    absolute_counts_df.rename(columns={'index': 'n-gram'}, inplace=True)

    # Fill NaN values with 0 (n-grams that don't appear in some genders)
    absolute_counts_df = absolute_counts_df.fillna(0)

    return absolute_counts_df


def select_features(absolute_counts_df: pd.DataFrame,
                   gender_distribution: pd.Series) -> List[str]:
    """
    Perform chi-square feature selection.

    Filters features based on:
    1. Statistical significance (p-value < threshold)
    2. Contains end token '<E>' (if FILTER_END_TOKEN_ONLY is True)

    Args:
        absolute_counts_df: DataFrame with n-gram counts from calculate_ngram_counts()
        gender_distribution: Series with expected gender proportions

    Returns:
        List of selected n-gram feature names

    Examples:
        >>> counts = pd.DataFrame({
        ...     'n-gram': ['<Se', 'ab<E>'],
        ...     'f': [100, 50],
        ...     'm': [50, 30],
        ...     'n': [30, 20]
        ... })
        >>> dist = pd.Series({'f': 0.5, 'm': 0.3, 'n': 0.2})
        >>> features = select_features(counts, dist)
        >>> all('<E>' in f for f in features)
        True
    """
    # Calculate p-values for each n-gram
    absolute_counts_df['P_Value'] = absolute_counts_df.apply(
        lambda row: calculate_p_value(row, gender_distribution),
        axis=1
    )

    # Sort by p-value
    sorted_df = absolute_counts_df.sort_values(by='P_Value', ascending=True)

    # Drop rows with NaN p-values (insufficient sample size)
    reduced_features = sorted_df.dropna()

    # Filter by p-value threshold
    reduced_features = reduced_features[reduced_features['P_Value'] < P_VALUE_THRESHOLD]

    # Filter to only n-grams containing end token (if configured)
    if FILTER_END_TOKEN_ONLY:
        reduced_features = reduced_features[
            reduced_features['n-gram'].apply(lambda x: END_TOKEN in x)
        ]

    # Return list of selected n-gram features
    selected_features = reduced_features['n-gram'].tolist()

    print(f"Reduced Feature Length: {len(selected_features)}")

    return selected_features


def create_feature_matrix(nouns: pd.DataFrame,
                         selected_features: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create one-hot encoded feature matrix for model training.

    Args:
        nouns: Original DataFrame with 'lemma' and 'genus' columns
        selected_features: List of selected n-gram features

    Returns:
        Tuple of (X, y):
        - X: DataFrame with boolean columns for each selected feature
        - y: Series with gender labels

    Examples:
        >>> nouns = pd.DataFrame({'lemma': ['Haus'], 'genus': ['n']})
        >>> features = ['s<E>', 'us<E>']
        >>> X, y = create_feature_matrix(nouns, features)
        >>> X.shape[1] == len(features)
        True
    """
    # Create new DataFrame
    feature_df = pd.DataFrame()

    # Add words with start and end tokens
    feature_df['Word'] = START_TOKEN + nouns['lemma'] + END_TOKEN

    # Add gender labels
    feature_df['Genus'] = nouns['genus']

    # Create one-hot encoding for each selected feature
    for feature in selected_features:
        feature_df[feature] = feature_df['Word'].apply(lambda word: feature in word)

    # Split into features (X) and labels (y)
    y = feature_df['Genus']
    X = feature_df.drop(columns=['Word', 'Genus'])

    return X, y


def prepare_training_data(nouns: pd.DataFrame,
                         gender_distribution: pd.Series) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Complete feature engineering pipeline.

    Performs all steps from n-gram generation to feature matrix creation.

    Args:
        nouns: DataFrame with 'lemma' and 'genus' columns
        gender_distribution: Series with expected gender proportions

    Returns:
        Tuple of (X, y, features):
        - X: Feature matrix (one-hot encoded n-grams)
        - y: Gender labels
        - features: List of selected feature names (needed for prediction)

    Examples:
        >>> nouns = pd.DataFrame({'lemma': ['Haus', 'Katze'], 'genus': ['n', 'f']})
        >>> dist = pd.Series({'f': 0.5, 'm': 0.3, 'n': 0.2})
        >>> X, y, features = prepare_training_data(nouns, dist)
        >>> len(features) > 0
        True
    """
    print("Generating n-grams...")
    nouns_with_ngrams = generate_ngrams(nouns)

    print("Calculating n-gram counts...")
    absolute_counts = calculate_ngram_counts(nouns_with_ngrams)

    print("Selecting features via chi-square test...")
    selected_features = select_features(absolute_counts, gender_distribution)

    print("Creating feature matrix...")
    X, y = create_feature_matrix(nouns, selected_features)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Training examples: {len(y)}")

    return X, y, selected_features
