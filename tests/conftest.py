"""
Pytest fixtures for germannouns tests.

Provides sample data, temporary files, and test utilities.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile


@pytest.fixture
def sample_nouns():
    """Sample German nouns DataFrame for testing."""
    return pd.DataFrame({
        'lemma': ['Haus', 'Katze', 'Mann', 'Frau', 'Kind'],
        'genus': ['n', 'f', 'm', 'f', 'n']
    })


@pytest.fixture
def sample_gender_distribution():
    """Sample gender distribution for testing."""
    return pd.Series({
        'f': 0.4,
        'm': 0.3,
        'n': 0.3
    })


@pytest.fixture
def sample_ngrams():
    """Sample DataFrame with n-grams."""
    return pd.DataFrame({
        'lemma': ['ab', 'cd'],
        'genus': ['f', 'm'],
        'Chs': [['<S>', 'a', 'b', '<E>'], ['<S>', 'c', 'd', '<E>']],
        'Chs2': [['<Sa', 'ab', 'b<E>'], ['<Sc', 'cd', 'd<E>']]
    })


@pytest.fixture
def sample_feature_matrix():
    """Sample feature matrix for testing."""
    X = pd.DataFrame({
        's<E>': [True, False, False],
        'n<E>': [False, True, False],
        'd<E>': [False, False, True]
    })
    y = pd.Series(['n', 'f', 'n'])
    features = ['s<E>', 'n<E>', 'd<E>']
    return X, y, features


@pytest.fixture
def temp_csv(tmp_path, sample_nouns):
    """Create temporary CSV file with sample nouns."""
    csv_path = tmp_path / "nouns.csv"
    sample_nouns.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def temp_model_file(tmp_path, sample_feature_matrix):
    """Create temporary model pickle file."""
    import pickle
    from sklearn.tree import DecisionTreeClassifier

    X, y, features = sample_feature_matrix

    # Train simple model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)

    # Save to temporary file
    model_path = tmp_path / "test_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump((model, features), f)

    return model_path
