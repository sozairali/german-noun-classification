"""
Thin CLI wrapper for gender prediction.

Usage:
    python scripts/predict.py Haus
    python scripts/predict.py  # Interactive mode
"""

from germannouns.model.predictor import main

if __name__ == "__main__":
    exit(main())
