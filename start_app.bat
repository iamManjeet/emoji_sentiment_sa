@echo off
REM Start script for Emoji Sentiment Analysis Web App (Windows)

echo 🚀 Starting Emoji Sentiment Analysis Web App...
echo ==============================================

REM Check if virtual environment exists
if not exist "venv" (
    echo ❌ Virtual environment not found. Please run setup first:
    echo    python setup.py
    pause
    exit /b 1
)

REM Activate virtual environment
echo 📦 Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if model exists
if not exist "models\logistic_emoji_True.joblib" (
    echo ❌ Trained model not found. Please train a model first:
    echo    python train.py --model_type logistic --use_emoji_features
    pause
    exit /b 1
)

REM Start Streamlit app
echo 🌐 Starting Streamlit web application...
echo    The app will be available at: http://localhost:8501
echo    Press Ctrl+C to stop the app
echo.

streamlit run app.py

pause
