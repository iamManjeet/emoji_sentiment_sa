#!/bin/bash
# Start script for Emoji Sentiment Analysis Web App

echo "🚀 Starting Emoji Sentiment Analysis Web App..."
echo "=============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first:"
    echo "   python setup.py"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if model exists
if [ ! -f "models/logistic_emoji_True.joblib" ]; then
    echo "❌ Trained model not found. Please train a model first:"
    echo "   python train.py --model_type logistic --use_emoji_features"
    exit 1
fi

# Start Streamlit app
echo "🌐 Starting Streamlit web application..."
echo "   The app will be available at: http://localhost:8501"
echo "   Press Ctrl+C to stop the app"
echo ""

streamlit run app.py
