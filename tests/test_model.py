"""
Tests for model training and prediction.

Tests model loading, prediction, and training utilities.
"""

import pytest
import pickle
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from germannouns.model.predictor import load_model, predict_gender
from germannouns.model.trainer import train_decision_tree, save_model


class TestLoadModel:
    """Test model loading."""

    def test_load_valid_model(self, temp_model_file):
        """Test loading a valid model."""
        model, features = load_model(temp_model_file)
        assert isinstance(model, DecisionTreeClassifier)
        assert isinstance(features, list)
        assert len(features) > 0

    def test_model_not_found(self, tmp_path):
        """Test error when model file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "nonexistent.pkl")

    def test_corrupted_model(self, tmp_path):
        """Test error with corrupted model file."""
        model_path = tmp_path / "corrupted.pkl"
        # Write invalid pickle data
        with open(model_path, 'w') as f:
            f.write("not a valid pickle file")

        with pytest.raises(Exception, match="Failed to load model"):
            load_model(model_path)

    def test_caching(self, temp_model_file):
        """Test that model is cached after first load."""
        # Clear cache first
        from germannouns.model import predictor
        predictor._model_cache = None

        # First load
        model1, features1 = load_model(temp_model_file)

        # Second load should return cached version
        model2, features2 = load_model(temp_model_file)

        # Should be the same objects
        assert model1 is model2
        assert features1 is features2


class TestPredictGender:
    """Test gender prediction."""

    def test_predict_valid_noun(self, temp_model_file):
        """Test prediction with valid noun."""
        # Clear cache
        from germannouns.model import predictor
        predictor._model_cache = None

        gender = predict_gender("test", temp_model_file)
        assert gender in ['die', 'der', 'das']

    def test_predict_empty_noun(self, temp_model_file):
        """Test error with empty noun."""
        with pytest.raises(ValueError, match="cannot be empty"):
            predict_gender("", temp_model_file)

    def test_predict_whitespace_only(self, temp_model_file):
        """Test error with whitespace-only noun."""
        with pytest.raises(ValueError, match="cannot be empty"):
            predict_gender("   ", temp_model_file)

    def test_predict_model_not_found(self, tmp_path):
        """Test error when model doesn't exist."""
        with pytest.raises(FileNotFoundError):
            predict_gender("Haus", tmp_path / "nonexistent.pkl")

    def test_predict_strips_whitespace(self, temp_model_file):
        """Test that whitespace is stripped from input."""
        # Clear cache
        from germannouns.model import predictor
        predictor._model_cache = None

        gender1 = predict_gender("test", temp_model_file)
        gender2 = predict_gender("  test  ", temp_model_file)
        assert gender1 == gender2


class TestTrainDecisionTree:
    """Test decision tree training."""

    def test_trains_successfully(self, sample_feature_matrix):
        """Test that model trains successfully."""
        X, y, features = sample_feature_matrix
        model = train_decision_tree(0.0001, X, y)
        assert isinstance(model, DecisionTreeClassifier)

    def test_can_predict(self, sample_feature_matrix):
        """Test that trained model can predict."""
        X, y, features = sample_feature_matrix
        model = train_decision_tree(0.0001, X, y)
        predictions = model.predict(X)
        assert len(predictions) == len(y)


class TestSaveModel:
    """Test model saving."""

    def test_saves_model(self, tmp_path, sample_feature_matrix):
        """Test that model is saved correctly."""
        X, y, features = sample_feature_matrix
        model = train_decision_tree(0.0001, X, y)

        model_path = tmp_path / "saved_model.pkl"
        save_model(model, features, model_path)

        assert model_path.exists()

    def test_saved_model_loads(self, tmp_path, sample_feature_matrix):
        """Test that saved model can be loaded."""
        X, y, features = sample_feature_matrix
        model = train_decision_tree(0.0001, X, y)

        model_path = tmp_path / "saved_model.pkl"
        save_model(model, features, model_path)

        # Load and verify
        with open(model_path, 'rb') as f:
            loaded_model, loaded_features = pickle.load(f)

        assert isinstance(loaded_model, DecisionTreeClassifier)
        assert loaded_features == features
