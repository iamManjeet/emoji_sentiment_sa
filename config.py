"""
Configuration file for emoji sentiment analysis project.
Contains all project settings, hyperparameters, and paths.
"""

from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
UTILS_DIR = BASE_DIR / "utils"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_CONFIG = {
    "sentiment140_url": "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip",
    "sentiment140_filename": "sentiment140.zip",
    "train_filename": "training.1600000.processed.noemoticon.csv",
    "test_filename": "testdata.manual.2009.06.14.csv",
    "columns": ['target', 'ids', 'date', 'query', 'user', 'text'],
    "label_mapping": {0: 0, 2: 2, 4: 1},  # 0=negative, 2=neutral, 4=positive
    "sample_size": 100000,  # Number of samples per class for faster training
    "test_size": 0.1,
    "val_size": 0.1,
    "random_state": 42
}

# Model configuration
MODEL_CONFIG = {
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
        "random_state": 42,
        "solver": "liblinear"
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42,
        "n_jobs": -1
    },
    "transformer": {
        "model_name": "distilbert-base-uncased",
        "num_epochs": 3,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "max_length": 128
    }
}

# Feature extraction configuration
FEATURE_CONFIG = {
    "tfidf": {
        "max_features": 10000,
        "ngram_range": (1, 2),
        "stop_words": "english",
        "min_df": 5,
        "max_df": 0.95
    },
    "emoji": {
        "use_emoji_features": True,
        "fallback_lexicon": {
            "🔥": 0.9, "😊": 0.8, "😁": 0.8, "😍": 0.9, "😎": 0.5, "👍": 0.6,
            "😢": -0.6, "😭": -0.8, "😡": -0.8, "👎": -0.6, "😴": -0.3, "🤮": -0.8
        }
    }
}

# Text preprocessing configuration
PREPROCESSING_CONFIG = {
    "remove_urls": True,
    "remove_mentions": True,
    "normalize_hashtags": True,
    "remove_numbers": True,
    "demojize": True,
    "lowercase": True,
    "remove_extra_spaces": True
}

# Evaluation configuration
EVALUATION_CONFIG = {
    "metrics": ["accuracy", "precision", "recall", "f1"],
    "cv_folds": 5,
    "random_state": 42
}

# Web application configuration
APP_CONFIG = {
    "page_title": "Emoji Sentiment Analysis",
    "page_icon": "😊",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "max_upload_size": 200 * 1024 * 1024,  # 200MB
    "supported_file_types": ["csv", "txt"]
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": BASE_DIR / "logs" / "app.log"
}

# Sentiment labels
SENTIMENT_LABELS = {
    0: "Negative",
    1: "Positive",
    2: "Neutral"
}

# Color schemes for visualization
COLOR_SCHEMES = {
    "sentiment": {
        "Positive": "#28a745",
        "Negative": "#dc3545",
        "Neutral": "#6c757d"
    },
    "gradient": {
        "start": "#667eea",
        "end": "#764ba2"
    }
}

# Emoji sentiment examples for testing
TEST_EXAMPLES = [
    "I love this movie! 😍",
    "This is terrible 😢",
    "It's okay 😐",
    "Amazing performance! 🔥",
    "I hate this 😡",
    "Great job! 👍",
    "This sucks 👎",
    "Boring 😴"
]
