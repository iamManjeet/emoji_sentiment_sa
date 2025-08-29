# 😊 Emoji Sentiment Analysis Project


A comprehensive machine learning project for sentiment analysis with special focus on emoji-based sentiment detection using the Sentiment140 dataset.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

<a id="project-overview"></a>
## 🎯 Project Overview

This project implements advanced sentiment analysis models that can understand and classify text sentiment, with particular attention to how emojis influence sentiment classification. The system uses both traditional NLP techniques and emoji-specific features to improve sentiment prediction accuracy.

### Key Highlights

- **Emoji-aware sentiment analysis**: Special handling of emoji sentiment using custom lexicon
- **Multiple model support**: Logistic Regression, Random Forest, and Transformer models
- **Web interface**: Beautiful Streamlit application with interactive visualizations
- **Production-ready**: Comprehensive error handling, testing, and documentation
- **High accuracy**: Achieves 76.85% accuracy on test data

<a id="features"></a>
## ✨ Features

### 🔍 Sentiment Analysis
- **Three-class classification**: Positive, Negative, Neutral
- **Emoji sentiment integration**: Custom lexicon with 12+ emojis
- **Confidence scores**: Probability estimates for predictions
- **Batch processing**: Handle multiple texts simultaneously

### 🛠️ Text Processing
- **URL normalization**: Converts URLs to `<url>` tokens
- **User mention handling**: Replaces @mentions with `<user>` tokens
- **Hashtag processing**: Splits camelCase hashtags into words
- **Emoji normalization**: Converts emojis to text representations
- **Number normalization**: Replaces numbers with `<num>` tokens

### 📊 Model Types
- **Logistic Regression**: Fast and interpretable
- **Random Forest**: Robust and handles non-linear patterns
- **Transformer Models**: State-of-the-art performance (DistilBERT)

### 🌐 Web Application
- **Interactive interface**: User-friendly Streamlit app
- **Real-time predictions**: Instant sentiment analysis
- **Visualizations**: Charts and graphs for results
- **File upload**: Support for CSV batch processing
- **Model selection**: Choose from different trained models

<a id="project-structure"></a>
## 📁 Project Structure

```
emoji_sentiment_sa/
├── 📁 data/                    # Dataset folder (placeholders kept)
│   ├── raw/                    # Raw dataset files (e.g., Sentiment140)
│   └── processed/              # Cleaned and split datasets
├── 📁 member1_data/            # Data pipeline scripts (placeholder)
├── 📁 models/                  # Pretrained model(s) used by the app
├── 📁 utils/                   # Core utilities
│   ├── emoji_lexicon.py        # Emoji sentiment lexicon handling
│   ├── preprocess.py           # Text preprocessing functions
│   └── metrics.py              # Evaluation metrics (optional)
├── 📁 venv/                    # Virtual environment (local)
├── 📄 app.py                   # Streamlit web application
├── 📄 train.py                 # Model class (used by the app to load models)
├── 📄 predict.py               # CLI predictor (optional)
├── 📄 requirements.txt         # Python dependencies
├── 📄 README.md                # This file
├── 📄 start_app.sh             # Easy start script (macOS/Linux)
└── 📄 start_app.bat            # Easy start script (Windows)
```

<a id="installation"></a>
## 🚀 Installation (Minimal)

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd emoji_sentiment_sa
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Ensure a model exists

Place a trained model file in `models/`, e.g. `models/logistic_emoji_True.joblib`.
If you don’t have one yet, you can use the included one (if provided) or train later.

<a id="quick-start"></a>
## ⚡ Quick Start

```bash
# Single text prediction (optional CLI)
python predict.py --text "I love this movie! 😍" --model_path models/logistic_emoji_True.joblib
```

### Run Web Application

**Option 1: Easy Start (Recommended)**
```bash
# On macOS/Linux:
./start_app.sh

# On Windows:
start_app.bat
```

**Option 2: Manual Start**
```bash
# Activate virtual environment first
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Start the app
streamlit run app.py
```

Open your browser and go to `http://localhost:8501`

<a id="usage"></a>
## 📖 Usage

### Command Line Interface (Optional)

```bash
# Single text prediction
python predict.py \
    --text "This movie is amazing! 🔥" \
    --model_path models/logistic_emoji_True.joblib

# Batch prediction from file
python predict.py \
    --file input_texts.csv \
    --model_path models/logistic_emoji_True.joblib \
    --output results.json

# Batch prediction with custom model
python predict.py \
    --file texts.txt \
    --model_path models/transformer_emoji_True \
    --model_type transformer
```

### Web Application

1. **Start the app**: `streamlit run app.py`
2. **Select a model** from the sidebar
3. **Choose analysis type**:
   - **Single Text**: Enter text and get instant results
   - **Batch Analysis**: Upload CSV file or enter multiple texts
   - **Model Info**: View model details and performance

### Programmatic Usage (Optional)

```python
from train import EmojiSentimentClassifier
from utils.preprocess import clean_tweet

# Load trained model
classifier = EmojiSentimentClassifier.load('models/logistic_emoji_True.joblib')

# Preprocess text
text = "I love this movie! 😍"
cleaned_text = clean_tweet(text)

# Make prediction
prediction = classifier.predict([cleaned_text])[0]
print(f"Sentiment: {prediction}")  # 1 for Positive, 0 for Negative
```

<a id="model-performance"></a>
## 📊 Model Performance

### Current Results (Logistic Regression with Emoji Features)

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 76.85% |
| **Macro F1 Score** | 76.84% |
| **Precision (Negative)** | 77.92% |
| **Recall (Negative)** | 74.94% |
| **Precision (Positive)** | 75.86% |
| **Recall (Positive)** | 78.76% |

### Model Comparison

| Model Type | Accuracy | Training Time | Memory Usage |
|------------|----------|---------------|--------------|
| Logistic Regression | 76.85% | ~2 minutes | Low |
| Random Forest | ~78% | ~10 minutes | Medium |
| Transformer | ~80% | ~30 minutes | High |

<a id="api-reference"></a>
## 🧩 API Reference

This project is primarily a web app + CLI. Public entry points:
- CLI: `predict.py` — arguments `--text`, `--file`, `--model_path`, `--model_type`, `--output`, `--emoji_only`
- Programmatic: `EmojiSentimentClassifier` in `train.py` with `load`, `predict`, `train`, and `extract_features`

For detailed usage, see the Usage and Programmatic Usage sections.

<a id="configuration"></a>
## 🔧 Configuration

The project uses a centralized configuration system in `config.py`:

```python
# Model configuration
MODEL_CONFIG = {
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
        "random_state": 42
    }
}

# Feature extraction
FEATURE_CONFIG = {
    "tfidf": {
        "max_features": 10000,
        "ngram_range": (1, 2),
        "stop_words": "english"
    }
}
```

<a id="testing"></a>
## 🧪 Testing (Optional)
You can create your own quick checks using small scripts or the CLI; built-in test harnesses were removed for a minimal app footprint.

<a id="example-usage"></a>
## 📝 Example Usage

### Text Preprocessing

```python
from utils.preprocess import clean_tweet

texts = [
    "I love this movie! 😍",
    "Check out https://example.com",
    "@user123 this is great! #awesome",
    "The price is $99.99"
]

for text in texts:
    cleaned = clean_tweet(text)
    print(f"Original: {text}")
    print(f"Cleaned:  {cleaned}\n")
```

### Emoji Sentiment Analysis

```python
from utils.emoji_lexicon import emoji_sentiment_feature, load_emoji_sentiment_csv

lexicon = load_emoji_sentiment_csv(None)
text = "I love this! 😍 This is terrible 😢"
score = emoji_sentiment_feature(text, lexicon)
print(f"Emoji sentiment score: {score:.3f}")
```

### Complete Workflow

```python
import pandas as pd
from train import EmojiSentimentClassifier
from utils.preprocess import clean_tweet

# Load model
classifier = EmojiSentimentClassifier.load('models/logistic_emoji_True.joblib')

# Analyze multiple texts
texts = [
    "I love this movie! 😍",
    "This is terrible 😢",
    "It's okay 😐"
]

results = []
for text in texts:
    cleaned = clean_tweet(text)
    prediction = classifier.predict([cleaned])[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    results.append({"text": text, "sentiment": sentiment})

df_results = pd.DataFrame(results)
print(df_results)
```

<a id="contributing"></a>
## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Run tests**: `python test_setup.py`
5. **Commit your changes**: `git commit -m 'Add amazing feature'`
6. **Push to the branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python test_setup.py

# Run linting (if you have flake8 installed)
flake8 .

# Run the web app in development mode
streamlit run app.py
```

<a id="license"></a>
## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Sentiment140 dataset** from Stanford University
- **Emoji sentiment lexicon** based on research by Kralj Novak et al.
- **Open-source libraries**: scikit-learn, transformers, streamlit, and many others
- **Community contributors** who helped improve this project

## 📞 Support

If you encounter any issues or have questions:

1. **Check the documentation** in this README
2. **Run the test suite**: `python test_setup.py`
3. **Check existing issues** on GitHub
4. **Create a new issue** with detailed information

## 🔄 Version History

- **v1.0.0**: Initial release with basic sentiment analysis
- **v1.1.0**: Added emoji sentiment features
- **v1.2.0**: Added web interface and transformer models
- **v1.3.0**: Improved documentation and testing


## 👤 About the Author

**Manjeet Singh** — Software Engineer | Learning Developer (Bengaluru)

- **Email**: iiammanjeet@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/manjeet2005/

### Summary
Passionate software engineer specializing in AI and cloud technologies, turning ideas into impactful software. Experienced in full‑stack development and team leadership, with active mentorship and open‑source contributions. Committed to continuous learning and practical application of cutting‑edge tech.



---

**Made with ❤️ for the NLP community**

---