# GermanNouns - German Noun Gender Classification

A machine learning system that predicts the grammatical gender (die/der/das) of German nouns based on character n-gram features.

## Features

- 🎯 **~73% accuracy** on test data using decision tree classifier
- 🔤 **N-gram feature extraction** - Analyzes character patterns (1-grams through 5-grams)
- 📊 **Chi-square feature selection** - Identifies statistically significant patterns
- 🌲 **Multiple models** - Decision tree, random forest, ensemble methods
- 🌐 **Web interface** - Simple web app for interactive predictions
- 🧪 **Comprehensive tests** - 40+ tests covering core functionality

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/sozairali/german-noun-classification.git
cd german-noun-classification

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Git Bash on Windows
# or: venv\Scripts\activate   # Command Prompt on Windows

# Install package
pip install -e .
```

### Train the Model

```bash
python scripts/train.py
```

This will:
1. Load ~200,000 German nouns from `nouns.csv`
2. Extract character n-grams with start/end tokens
3. Select features using chi-square test (p < 0.05)
4. Train decision tree with cost complexity pruning
5. Save model to `decision_tree_ccp_model.pkl`

Training takes ~5-10 minutes depending on your machine.

### Make Predictions

```bash
# Single prediction
python scripts/predict.py Haus
# Output: das Haus

# Interactive mode
python scripts/predict.py
# Enter noun: Katze
# → die Katze
# Enter noun: Mann
# → der Mann
```

### Web Interface

```bash
cd web
python app.py
# Visit http://localhost:5000
```

## How It Works

### 1. Feature Extraction

The system analyzes character patterns in German words using n-grams:

```python
Word: "Haus"
1-grams: <S>, H, a, u, s, <E>
2-grams: <SH, Ha, au, us, s<E>
3-grams: <SHa, Hau, aus, us<E>
...
```

The start token `<S>` and end token `<E>` mark word boundaries.

### 2. Feature Selection

Chi-square test identifies n-grams with significant gender associations:

- Tests null hypothesis: "n-gram distribution is independent of gender"
- Keeps features with p-value < 0.05 (statistically significant)
- Filters to n-grams containing `<E>` (word endings are most predictive)

Example significant features:
- `ung<E>` → feminine (die Zeitung, die Wohnung)
- `er<E>` → masculine (der Lehrer, der Computer)
- `chen<E>` → neuter (das Mädchen, das Häuschen)

### 3. Model Training

Decision tree classifier with cost complexity pruning:

```
Training data: 60% (~120k nouns)
Test data: 20% (~40k nouns)
Validation data: 20% (~40k nouns)
```

The model learns rules like:
```
if word contains 'ung<E>':
    predict 'die' (feminine)
else if word contains 'er<E>':
    predict 'der' (masculine)
...
```

### 4. Performance

| Metric | Value |
|--------|-------|
| Training accuracy | ~75% |
| Test accuracy | ~73% |
| Features selected | ~1000 n-grams |
| Tree depth | ~40 nodes |

## Project Structure

```
germannouns/
├── src/germannouns/           # Main package
│   ├── config.py              # Configuration constants
│   ├── data/                  # Data loading and feature engineering
│   ├── model/                 # Training and prediction
│   └── utils/                 # Shared utilities
├── scripts/                   # CLI entry points
│   ├── train.py              # Model training
│   └── predict.py            # Prediction
├── web/                       # Flask web app
│   ├── app.py
│   ├── templates/
│   └── static/
├── tests/                     # Test suite
├── nouns.csv                  # Training data
├── decision_tree_ccp_model.pkl  # Trained model
└── pyproject.toml             # Package metadata
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=germannouns --cov-report=html
```

### Code Style

The project follows these principles:
- **Type hints** on all functions
- **Docstrings** in Google style
- **Single responsibility** - Each module has one clear purpose
- **Configuration centralized** in `config.py`
- **Error handling** with clear messages

## Data

The `nouns.csv` dataset contains ~200,000 German nouns from the German language corpus:

| Column | Description | Example |
|--------|-------------|---------|
| lemma | German noun | "Haus" |
| genus | Gender code | "n" (neuter) |

Gender codes:
- `f` - Feminine (die)
- `m` - Masculine (der)
- `n` - Neuter (das)

Distribution:
- Feminine: ~40%
- Masculine: ~30%
- Neuter: ~30%

## Technical Details

### N-gram Generation

```python
from germannouns.utils.core import create_ngram

# Generate bigrams
create_ngram("Haus", 2)
# Returns: ['<SH', 'Ha', 'au', 'us', 's<E>']
```

### Chi-Square Test

For each n-gram, test independence from gender:

```
H0: P(n-gram | gender) = P(n-gram)
H1: P(n-gram | gender) ≠ P(n-gram)

If p-value < 0.05: reject H0 (feature is significant)
```

### Model Persistence

The trained model is saved with its feature list:

```python
import pickle

# Save
with open('decision_tree_ccp_model.pkl', 'wb') as f:
    pickle.dump((model, features), f)

# Load
with open('decision_tree_ccp_model.pkl', 'rb') as f:
    model, features = pickle.load(f)
```

## API Reference

### Core Functions

```python
from germannouns.utils.core import create_ngram, calculate_p_value
from germannouns.model.predictor import predict_gender

# Generate n-grams
ngrams = create_ngram("Haus", n=3)

# Predict gender
article = predict_gender("Haus")  # Returns: "das"
```

### Training Pipeline

```python
from germannouns.data.loader import load_nouns
from germannouns.data.feature_engineering import prepare_training_data
from germannouns.model.trainer import train_model

# Load data
nouns = load_nouns()

# Prepare features
X, y, features = prepare_training_data(nouns, gender_dist)

# Train model
model = train_model(X, y, features)
```

## Deployment

The web app can be deployed to Heroku or any platform supporting Python:

```bash
# Procfile
web: gunicorn web.app:app

# runtime.txt
python-3.11.0
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass (`pytest tests/`)
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Dataset: German language corpus
- Inspiration: Morphological analysis of German nouns
- Pattern: Lantbot package structure

## Contact

- GitHub: [@sozairali](https://github.com/sozairali)
- Repository: [german-noun-classification](https://github.com/sozairali/german-noun-classification)
