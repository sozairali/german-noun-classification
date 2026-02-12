"""
Tests for core utility functions.

Tests n-gram generation, chi-square testing, and data transformation utilities.
"""

import pytest
import pandas as pd
import numpy as np
from germannouns.utils.core import (
    create_ngram,
    calculate_p_value,
    create_dictsof_counts,
    convert_dicts_to_df
)


class TestCreateNgram:
    """Test n-gram generation."""

    def test_unigram(self):
        """Test unigram (n=1) generation."""
        result = create_ngram("ab", 1)
        assert result == ['<S>', 'a', 'b', '<E>']

    def test_bigram(self):
        """Test bigram (n=2) generation."""
        result = create_ngram("ab", 2)
        assert result == ['<Sa', 'ab', 'b<E>']

    def test_trigram(self):
        """Test trigram (n=3) generation."""
        result = create_ngram("ab", 3)
        assert result == ['<Sab', 'ab<E>']

    def test_fourgram(self):
        """Test 4-gram generation."""
        result = create_ngram("ab", 4)
        assert result == ['<Sab<E>']

    def test_empty_string(self):
        """Test empty string handling."""
        result = create_ngram("", 1)
        assert result == ['<S>', '<E>']

    def test_single_character(self):
        """Test single character word."""
        result = create_ngram("a", 1)
        assert result == ['<S>', 'a', '<E>']

        result = create_ngram("a", 2)
        assert result == ['<Sa', 'a<E>']

    def test_longer_word(self):
        """Test longer word with trigrams."""
        result = create_ngram("Haus", 3)
        expected = ['<SHa', 'Hau', 'aus', 'us<E>']
        assert result == expected


class TestCalculatePValue:
    """Test chi-square p-value calculation."""

    def test_valid_distribution(self, sample_gender_distribution):
        """Test with valid distribution."""
        row = pd.Series({
            'f': 100,
            'm': 60,
            'n': 40,
            'n-gram': 'test'
        })
        p_value = calculate_p_value(row, sample_gender_distribution)
        assert isinstance(p_value, float)
        assert 0.0 <= p_value <= 1.0

    def test_low_counts_returns_nan(self, sample_gender_distribution):
        """Test that low counts return NaN."""
        row = pd.Series({
            'f': 3,
            'm': 2,
            'n': 1,
            'n-gram': 'test'
        })
        p_value = calculate_p_value(row, sample_gender_distribution)
        assert pd.isna(p_value)

    def test_perfectly_distributed(self):
        """Test perfectly distributed data (should have high p-value)."""
        expected = pd.Series({'f': 0.4, 'm': 0.3, 'n': 0.3})
        row = pd.Series({
            'f': 400,
            'm': 300,
            'n': 300,
            'n-gram': 'test'
        })
        p_value = calculate_p_value(row, expected)
        assert p_value > 0.9  # Should be close to 1.0


class TestCreateDictsOfCounts:
    """Test counting n-grams by category."""

    def test_simple_count(self, sample_ngrams):
        """Test simple counting."""
        result = create_dictsof_counts(
            sample_ngrams,
            sample_ngrams['Chs'],
            'genus'
        )
        # '<S>' appears once in 'f' and once in 'm'
        assert result['<S>'] == {'f': 1, 'm': 1}
        # 'a' appears once in 'f' only
        assert result['a'] == {'f': 1}
        # 'c' appears once in 'm' only
        assert result['c'] == {'m': 1}

    def test_multiple_occurrences(self):
        """Test counting with multiple occurrences."""
        df = pd.DataFrame({
            'genus': ['f', 'f', 'm'],
            'ngrams': [['ab'], ['ab'], ['ab']]
        })
        result = create_dictsof_counts(df, df['ngrams'], 'genus')
        assert result['ab'] == {'f': 2, 'm': 1}


class TestConvertDictsToDF:
    """Test converting nested dicts to DataFrame."""

    def test_single_dict(self):
        """Test converting single dictionary."""
        dict1 = {'ab': {'f': 10, 'm': 5}}
        result = convert_dicts_to_df([dict1])
        assert 'ab' in result.index
        assert result.loc['ab', 'f'] == 10
        assert result.loc['ab', 'm'] == 5

    def test_multiple_dicts(self):
        """Test converting multiple dictionaries."""
        dict1 = {'ab': {'f': 10, 'm': 5}}
        dict2 = {'abc': {'f': 3, 'm': 2, 'n': 1}}
        result = convert_dicts_to_df([dict1, dict2])
        assert 'ab' in result.index
        assert 'abc' in result.index

    def test_empty_list(self):
        """Test empty list of dictionaries."""
        result = convert_dicts_to_df([])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
