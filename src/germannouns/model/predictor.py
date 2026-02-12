"""
Gender prediction for German nouns.

Loads trained model and predicts the appropriate article (die/der/das)
for a given German noun.
"""

import pickle
from typing import Tuple, List, Optional
from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from germannouns.config import MODEL_PICKLE, GENDER_MAPPING, START_TOKEN, END_TOKEN


# Global cache for model and features (lazy loading)
_model_cache: Optional[Tuple[DecisionTreeClassifier, List[str]]] = None


def load_model(model_path: Path = MODEL_PICKLE) -> Tuple[DecisionTreeClassifier, List[str]]:
    """
    Load trained model and feature list from pickle file.

    Uses caching to avoid reloading model on subsequent calls.

    Args:
        model_path: Path to model pickle file

    Returns:
        Tuple of (model, features):
        - model: Trained DecisionTreeClassifier
        - features: List of n-gram feature names

    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If pickle file is corrupted or incompatible

    Examples:
        >>> model, features = load_model()
        >>> isinstance(model, DecisionTreeClassifier)
        True
        >>> len(features) > 0
        True
    """
    global _model_cache

    # Return cached model if available
    if _model_cache is not None:
        return _model_cache

    # Check if model file exists
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model file not found at: {model_path}\n"
            f"Please train the model first using: python scripts/train.py"
        )

    # Load model and features from pickle
    try:
        with open(model_path, 'rb') as file:
            model, features = pickle.load(file)

        # Validate loaded data
        if not isinstance(features, list) or len(features) == 0:
            raise ValueError("Invalid features list in model file")

        # Cache for future use
        _model_cache = (model, features)

        return model, features

    except Exception as e:
        raise Exception(
            f"Failed to load model from {model_path}. "
            f"The file may be corrupted or incompatible. Error: {e}"
        )


def predict_gender(noun: str, model_path: Path = MODEL_PICKLE) -> str:
    """
    Predict the gender article (die/der/das) for a German noun.

    Args:
        noun: The German noun (e.g., "Haus", "Katze", "Mann")
        model_path: Path to model pickle file (optional, uses default)

    Returns:
        The appropriate article: "die" (feminine), "der" (masculine), or "das" (neuter)

    Raises:
        ValueError: If noun is empty or contains only whitespace
        FileNotFoundError: If model file doesn't exist
        Exception: If prediction fails

    Examples:
        >>> predict_gender("Haus")
        'das'
        >>> predict_gender("Katze")
        'die'
        >>> predict_gender("Mann")
        'der'
    """
    # Validate input
    if not noun or not noun.strip():
        raise ValueError("Noun cannot be empty or whitespace")

    # Load model and features
    model, features = load_model(model_path)

    # Prepare noun with start and end tokens
    noun_tagged = f'{START_TOKEN}{noun.strip()}{END_TOKEN}'

    # Create feature vector (one-hot encoding)
    feature_vector = {col: bool(col in noun_tagged) for col in features}
    feature_df = pd.DataFrame(feature_vector, index=[0])

    # Predict gender code
    try:
        gender_code = model.predict(feature_df)[0]
    except Exception as e:
        raise Exception(f"Prediction failed for noun '{noun}': {e}")

    # Map gender code to article
    article = GENDER_MAPPING.get(gender_code)

    if article is None:
        raise ValueError(
            f"Unknown gender code '{gender_code}' predicted for noun '{noun}'. "
            f"Expected one of: {list(GENDER_MAPPING.keys())}"
        )

    return article


def main():
    """
    Main entry point for prediction script.

    Allows interactive testing of the model or single predictions.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict German noun gender (die/der/das)"
    )
    parser.add_argument(
        "noun",
        nargs="?",
        help="German noun to predict (if omitted, enters interactive mode)"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PICKLE,
        help=f"Path to model file (default: {MODEL_PICKLE})"
    )

    args = parser.parse_args()

    # Single prediction mode
    if args.noun:
        try:
            article = predict_gender(args.noun, args.model)
            print(f"{article} {args.noun}")
        except Exception as e:
            print(f"Error: {e}")
            return 1

    # Interactive mode
    else:
        print("German Noun Gender Predictor")
        print("=" * 40)
        print("Enter German nouns to predict their gender.")
        print("Type 'quit' or 'exit' to stop.\n")

        try:
            # Load model once
            load_model(args.model)
            print("Model loaded successfully!\n")
        except Exception as e:
            print(f"Error loading model: {e}")
            return 1

        while True:
            try:
                noun = input("Enter noun: ").strip()

                if noun.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if not noun:
                    continue

                article = predict_gender(noun, args.model)
                print(f"→ {article} {noun}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")

    return 0


if __name__ == "__main__":
    exit(main())
