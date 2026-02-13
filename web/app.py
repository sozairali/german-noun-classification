"""
Flask web application for German noun gender prediction.

Provides a simple web interface for predicting German noun articles (die/der/das).
"""

import os
import sys
import logging
from pathlib import Path
from flask import Flask, request, jsonify, render_template

# Add parent directory to path to import germannouns package
# This allows web app to work both standalone and when package is installed
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    from src.germannouns.model.predictor import predict_gender
except ImportError:
    # Try alternative import if package is installed
    from germannouns.model.predictor import predict_gender

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@app.route('/')
def homepage():
    """Render the homepage with noun input form."""
    return render_template("index.html")


@app.route('/predict_gender/<noun>')
def predict_gender_route(noun):
    """
    Predict gender for a given noun.

    Args:
        noun: German noun from URL path

    Returns:
        JSON response with noun and predicted gender article

    Examples:
        GET /predict_gender/Haus
        → {"noun": "Haus", "gender": "das"}
    """
    try:
        # Validate input
        if not noun or not noun.strip():
            return jsonify({'error': 'Noun cannot be empty'}), 400

        # Strip whitespace
        noun = noun.strip()

        # Log request
        logger.info(f"Predicting gender for: {noun}")

        # Predict gender
        gender = predict_gender(noun)

        # Return response
        return jsonify({'noun': noun, 'gender': gender})

    except ValueError as e:
        # Input validation errors
        logger.warning(f"Validation error for '{noun}': {e}")
        return jsonify({'error': str(e)}), 400

    except FileNotFoundError as e:
        # Model not found
        logger.error(f"Model not found: {e}")
        return jsonify({'error': 'Model not trained. Please train the model first.'}), 500

    except Exception as e:
        # Unexpected errors
        logger.error(f"Prediction error for '{noun}': {e}")
        return jsonify({'error': 'Prediction failed. Please try again.'}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Get host and port from environment variables
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))

    # Check if model exists
    try:
        # Try to load model to verify it exists
        from src.germannouns.model.predictor import load_model
        load_model()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.warning(f"Could not load model: {e}")
        logger.warning("The app will start, but predictions will fail until model is trained.")

    # Start Flask app
    logger.info(f"Starting Flask app on {host}:{port}")
    app.run(host=host, port=port, debug=False)
