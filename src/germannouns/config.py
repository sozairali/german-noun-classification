"""
Configuration for German noun gender classification.
All magic numbers and parameters consolidated here.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT
NOUNS_CSV = PROJECT_ROOT / "nouns.csv"
MODEL_PICKLE = PROJECT_ROOT / "decision_tree_ccp_model.pkl"

# Gender categories
GENDERS = ['f', 'm', 'n']
GENDER_MAPPING = {
    'f': 'die',
    'm': 'der',
    'n': 'das'
}

# N-gram configuration
NGRAM_RANGE = (1, 6)  # Create 1-grams through 5-grams (range is 1 to 6 exclusive)
NGRAM_COLUMN_NAMES = {1: 'Chs', 2: 'Chs2', 3: 'Chs3', 4: 'Chs4', 5: 'Chs5', 6: 'Chs6'}
START_TOKEN = '<S>'
END_TOKEN = '<E>'

# Feature engineering
P_VALUE_THRESHOLD = 0.05  # Chi-square significance level
MIN_NGRAM_COUNT = 5       # Minimum expected count for chi-square test (from calculate_p_value)
FILTER_END_TOKEN_ONLY = True  # Only keep n-grams containing '<E>'

# Model training
TRAIN_TEST_SPLIT = 0.4
VALIDATION_SPLIT = 0.5
RANDOM_STATE = 20

# Decision tree (post-pruning)
CCP_ALPHA = 0.0001

# Decision tree (pre-pruning via GridSearch)
GRID_SEARCH_PARAMS = {
    'max_depth': [*range(10, 20)],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2]
}

# Random forest
RF_GRID_SEARCH_PARAMS = {
    'n_estimators': [200, 300, 400],
    'max_depth': [20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['log2', 'sqrt']
}
RF_RANDOM_SEARCH_ITERATIONS = 100
RF_RANDOM_STATE = 42
RF_CV_FOLDS = 5

# Parallel processing
N_JOBS = -1  # Use all available cores

# Visualization
DEFAULT_PLOT_SIZE = 50
