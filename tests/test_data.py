"""
Tests for data loading and feature engineering.

Tests data loading, cleaning, n-gram generation, and feature selection.
"""

import pytest
import pandas as pd
from pathlib import Path
from germannouns.data.loader import load_nouns, get_gender_distribution
from germannouns.data.feature_engineering import (
    generate_ngrams,
    calculate_ngram_counts,
    select_features,
    create_feature_matrix
)


class TestLoadNouns:
    """Test noun data loading and cleaning."""

    def test_load_valid_csv(self, temp_csv):
        """Test loading valid CSV file."""
        nouns = load_nouns(temp_csv)
        assert isinstance(nouns, pd.DataFrame)
        assert 'lemma' in nouns.columns
        assert 'genus' in nouns.columns
        assert len(nouns) == 5

    def test_file_not_found(self, tmp_path):
        """Test error when CSV doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_nouns(tmp_path / "nonexistent.csv")

    def test_missing_columns(self, tmp_path):
        """Test error when required columns are missing."""
        csv_path = tmp_path / "bad.csv"
        df = pd.DataFrame({'wrong': [1, 2, 3]})
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="must contain"):
            load_nouns(csv_path)

    def test_special_character_removal(self, tmp_path):
        """Test removal of nouns starting with special characters."""
        csv_path = tmp_path / "nouns_with_special.csv"
        df = pd.DataFrame({
            'lemma': ['Haus', '$bad', '123num', 'Katze', '!also_bad'],
            'genus': ['n', 'f', 'm', 'f', 'n']
        })
        df.to_csv(csv_path, index=False)

        nouns = load_nouns(csv_path)
        assert len(nouns) == 2  # Only 'Haus' and 'Katze' remain
        assert 'Haus' in nouns['lemma'].values
        assert 'Katze' in nouns['lemma'].values

    def test_missing_value_removal(self, tmp_path):
        """Test removal of rows with missing values."""
        csv_path = tmp_path / "nouns_with_nan.csv"
        df = pd.DataFrame({
            'lemma': ['Haus', None, 'Katze'],
            'genus': ['n', 'f', None]
        })
        df.to_csv(csv_path, index=False)

        nouns = load_nouns(csv_path)
        assert len(nouns) == 1  # Only 'Haus' remains


class TestGetGenderDistribution:
    """Test gender distribution calculation."""

    def test_distribution(self, sample_nouns):
        """Test distribution calculation."""
        dist = get_gender_distribution(sample_nouns)
        assert isinstance(dist, pd.Series)
        assert 'f' in dist.index
        assert 'm' in dist.index
        assert 'n' in dist.index
        assert dist.sum() == pytest.approx(1.0)

    def test_distribution_values(self, sample_nouns):
        """Test distribution values are correct."""
        dist = get_gender_distribution(sample_nouns)
        # sample_nouns has 2 'f', 1 'm', 2 'n'
        assert dist['f'] == 0.4
        assert dist['m'] == 0.2
        assert dist['n'] == 0.4


class TestGenerateNgrams:
    """Test n-gram generation."""

    def test_generates_ngram_columns(self, sample_nouns):
        """Test that n-gram columns are created."""
        result = generate_ngrams(sample_nouns)
        assert 'Chs' in result.columns
        assert 'Chs2' in result.columns
        assert 'Chs3' in result.columns
        assert 'Chs4' in result.columns
        assert 'Chs5' in result.columns

    def test_ngram_content(self, sample_nouns):
        """Test n-gram content is correct."""
        result = generate_ngrams(sample_nouns)
        # Check first word 'Haus'
        haus_unigrams = result.iloc[0]['Chs']
        assert '<S>' in haus_unigrams
        assert 'H' in haus_unigrams
        assert '<E>' in haus_unigrams

    def test_original_columns_preserved(self, sample_nouns):
        """Test that original columns are preserved."""
        result = generate_ngrams(sample_nouns)
        assert 'lemma' in result.columns
        assert 'genus' in result.columns


class TestCalculateNgramCounts:
    """Test n-gram counting."""

    def test_count_output_format(self, sample_ngrams):
        """Test output format."""
        result = calculate_ngram_counts(sample_ngrams)
        assert isinstance(result, pd.DataFrame)
        assert 'n-gram' in result.columns
        assert 'f' in result.columns
        assert 'm' in result.columns

    def test_count_values(self, sample_ngrams):
        """Test count values are correct."""
        result = calculate_ngram_counts(sample_ngrams)
        # '<S>' appears in both 'f' and 'm'
        start_row = result[result['n-gram'] == '<S>'].iloc[0]
        assert start_row['f'] == 1
        assert start_row['m'] == 1


class TestSelectFeatures:
    """Test feature selection."""

    def test_filters_by_end_token(self, sample_gender_distribution):
        """Test that only features with end token are selected."""
        counts = pd.DataFrame({
            'n-gram': ['<Se', 'ab', 'b<E>'],
            'f': [100, 50, 30],
            'm': [50, 30, 20],
            'n': [30, 20, 10]
        })
        features = select_features(counts, sample_gender_distribution)
        # Only n-grams with '<E>' should be selected
        assert all('<E>' in f for f in features)

    def test_returns_list(self, sample_gender_distribution):
        """Test that a list is returned."""
        counts = pd.DataFrame({
            'n-gram': ['a<E>', 'b<E>'],
            'f': [100, 50],
            'm': [50, 30],
            'n': [30, 20]
        })
        features = select_features(counts, sample_gender_distribution)
        assert isinstance(features, list)


class TestCreateFeatureMatrix:
    """Test feature matrix creation."""

    def test_matrix_shape(self, sample_nouns):
        """Test matrix shape is correct."""
        features = ['s<E>', 'n<E>']
        X, y = create_feature_matrix(sample_nouns, features)
        assert X.shape == (5, 2)  # 5 nouns, 2 features
        assert len(y) == 5

    def test_one_hot_encoding(self):
        """Test one-hot encoding is correct."""
        nouns = pd.DataFrame({
            'lemma': ['Haus'],
            'genus': ['n']
        })
        features = ['s<E>', 'us<E>']
        X, y = create_feature_matrix(nouns, features)

        # 'Haus' tagged as '<SHaus<E>' contains both 's<E>' and 'us<E>'
        assert X.iloc[0]['s<E>'] == True
        assert X.iloc[0]['us<E>'] == True

    def test_labels_correct(self, sample_nouns):
        """Test that labels are correctly extracted."""
        features = ['s<E>']
        X, y = create_feature_matrix(sample_nouns, features)
        assert y.tolist() == ['n', 'f', 'm', 'f', 'n']
