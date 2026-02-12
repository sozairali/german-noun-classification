"""
germannouns - German noun gender classification using machine learning.

Predicts German noun gender (die/der/das) based on character n-gram features
using decision trees and random forests.
"""

__version__ = "0.1.0"

from germannouns.model.predictor import predict_gender

__all__ = ["predict_gender"]
