#!/usr/bin/env python3
"""
Streamlit web application for emoji sentiment analysis.
Provides an interactive interface for sentiment prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import time
import joblib

from train import EmojiSentimentClassifier
from utils.preprocess import clean_tweet
from utils.emoji_lexicon import emoji_sentiment_feature, load_emoji_sentiment_csv
import re

# Setup paths
BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"

# Page configuration
st.set_page_config(
    page_title="Emoji Sentiment Analysis",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #6c757d;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path):
    """Load the trained model with caching."""
    try:
        # Load the model directly like in predict.py
        saved_data = joblib.load(model_path)
        
        # Create classifier instance
        classifier = EmojiSentimentClassifier(
            model_type=saved_data['model_type'],
            use_emoji_features=saved_data['use_emoji_features']
        )
        classifier.model = saved_data['model']
        classifier.vectorizer = saved_data['vectorizer']
        classifier.emoji_lexicon = saved_data['emoji_lexicon']
        
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_sentiment_color(sentiment):
    """Get color for sentiment display."""
    colors = {
        "Positive": "#28a745",
        "Negative": "#dc3545", 
        "Neutral": "#6c757d"
    }
    return colors.get(sentiment, "#6c757d")

def get_sentiment_emoji(sentiment):
    """Get emoji for sentiment display."""
    emojis = {
        "Positive": "😊",
        "Negative": "😢",
        "Neutral": "😐"
    }
    return emojis.get(sentiment, "🤔")

def analyze_text(text, classifier):
    """Analyze text sentiment and return detailed results."""
    if not text.strip():
        return None
    
    # Preprocess text
    cleaned_text = clean_tweet(text)
    
    # Get emoji sentiment if available
    emoji_score = 0
    if hasattr(classifier, 'emoji_lexicon') and classifier.emoji_lexicon:
        emoji_score = emoji_sentiment_feature(text, classifier.emoji_lexicon)
    
    # Make prediction with heuristic adjustment similar to CLI
    try:
        features = classifier.extract_features([cleaned_text])
        if hasattr(classifier.model, 'predict_proba'):
            proba = classifier.model.predict_proba(features)[0]

            # Heuristic: downweight positive if negative emoji and negation cues present
            cleaned_lower = cleaned_text.lower()
            negated_negative_pattern = re.compile(r"\bnot\s+(that\s+|so\s+|very\s+)?(bad|terrible|awful|poor|boring|worse)\b")
            angry_emojis = {"😡", "😠", "🤬"}
            positive_emojis = {"😊", "😁", "😍", "👍", "🔥", "😎"}
            negative_emojis = {"😡", "😠", "🤬", "😢", "😭", "👎", "🤮", "😞", "😒", "☹️", "😕", "😔", "😩", "😫"}
            negative_cues = (
                ' not ', ' no ', "n't", ' never ', ' bad', ' terrible', ' awful', ' worst',
                ' poor', ' hate', ' dislike', ' sucks', ' disappointed', ' boring'
            )
            has_negative_cue = any(cue in f' {cleaned_lower} ' for cue in negative_cues) or 'not good' in cleaned_lower or 'no good' in cleaned_lower
            if bool(negated_negative_pattern.search(cleaned_lower)):
                pos_p = float(proba[1])
                neg_p = float(proba[0])
                desired_pos = max(pos_p, neg_p + 0.25)
                desired_pos = min(0.95, desired_pos)
                desired_neg = 1.0 - desired_pos
                proba = np.array([desired_neg, desired_pos])
            # Emoji-only mode: decide from emojis when present
            if st.session_state.get('emoji_only_mode'):
                has_pos = any(em in text for em in positive_emojis)
                has_neg = any(em in text for em in negative_emojis)
                if has_pos or has_neg:
                    if has_pos and not has_neg:
                        proba = np.array([0.05, 0.95])
                    elif has_neg and not has_pos:
                        proba = np.array([0.95, 0.05])
                    else:
                        proba = np.array([0.65, 0.35]) if emoji_score < 0 else np.array([0.35, 0.65])

            # Angry emoji bias (unless negated-negative present)
            if any(em in text for em in angry_emojis) and not bool(negated_negative_pattern.search(cleaned_lower)):
                pos_p = float(proba[1])
                neg_p = float(proba[0])
                desired_neg = max(neg_p, pos_p + 0.2)
                desired_neg = min(0.95, desired_neg)
                desired_pos = 1.0 - desired_neg
                proba = np.array([desired_neg, desired_pos])
            if emoji_score <= -0.35 and has_negative_cue:
                pos_p = float(proba[1])
                neg_p = float(proba[0])
                gap = max(0.0, pos_p - neg_p)
                shift = min(0.7, max(0.4, abs(emoji_score))) * gap
                pos_p = max(0.0, pos_p - shift)
                neg_p = min(1.0, neg_p + shift)
                if (pos_p - neg_p) < 0.15 and abs(emoji_score) >= 0.5:
                    extra = min(0.1, 0.5 * (0.15 - (pos_p - neg_p)))
                    pos_p = max(0.0, pos_p - extra)
                    neg_p = min(1.0, neg_p + extra)
                proba = np.array([neg_p, pos_p])

            # Additional negation pattern: text-only (e.g., "not ... good")
            negation_positive_pattern = re.compile(r"\bnot\s+(a\s+)?(really\s+|very\s+|that\s+|so\s+)?(good|great|amazing|awesome|nice|love|like)\b")
            explicit_negative_pattern = re.compile(r"\b(bad|terrible|awful|poor|worse|worst|hate|dislike|boring|sucks|disappointed)\b")
            has_negated_positive = bool(negation_positive_pattern.search(cleaned_lower))
            has_explicit_negative = bool(explicit_negative_pattern.search(cleaned_lower))
            if (has_negated_positive or has_explicit_negative) and not bool(negated_negative_pattern.search(cleaned_lower)):
                pos_p = float(proba[1])
                neg_p = float(proba[0])
                gap = max(0.0, pos_p - neg_p)
                base = 0.55 if has_negated_positive else 0.4
                shift = min(0.8, base + 0.2 * (1.0 if has_explicit_negative else 0.0)) * (gap if gap > 0 else 0.25)
                pos_p = max(0.0, pos_p - shift)
                neg_p = min(1.0, neg_p + shift)
                proba = np.array([neg_p, pos_p])

            prediction = int(np.argmax(proba))
            confidence = float(np.max(proba))
        else:
            prediction = classifier.predict([cleaned_text])[0]
            confidence = None
        sentiment_labels = {0: "Negative", 1: "Positive", 2: "Neutral"}
        sentiment = sentiment_labels.get(prediction, "Unknown")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None
    
    return {
        'text': text,
        'cleaned_text': cleaned_text,
        'sentiment': sentiment,
        'prediction': prediction,
        'confidence': confidence,
        'emoji_score': emoji_score
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">😊 Emoji Sentiment Analysis</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown("---")
    # Emoji-only toggle persisted in session
    if 'emoji_only_mode' not in st.session_state:
        st.session_state['emoji_only_mode'] = False
    st.session_state['emoji_only_mode'] = st.sidebar.toggle(
        "Emoji-only mode",
        value=st.session_state['emoji_only_mode'],
        help="If enabled, sentiment is decided from emojis when present."
    )
    
    # Model selection
    model_files = list(MODELS.glob("*.joblib")) + list(MODELS.glob("transformer*"))
    if not model_files:
        st.error("No trained models found. Please train a model first using `python train.py`")
        return
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=model_files,
        format_func=lambda x: x.name
    )
    
    # Load model
    classifier = load_model(str(selected_model))
    if classifier is None:
        return
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📝 Single Text Analysis", "📊 Batch Analysis", "📈 Model Info"])
    
    with tab1:
        st.header("Single Text Sentiment Analysis")
        
        # Text input
        text_input = st.text_area(
            "Enter text to analyze:",
            placeholder="Type your text here... (e.g., 'I love this movie! 😍')",
            height=150
        )
        
        if st.button("🔍 Analyze Sentiment", type="primary"):
            if text_input.strip():
                with st.spinner("Analyzing sentiment..."):
                    result = analyze_text(text_input, classifier)
                
                if result:
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Sentiment",
                            f"{get_sentiment_emoji(result['sentiment'])} {result['sentiment']}",
                            delta=None
                        )
                    
                    with col2:
                        if result['confidence']:
                            st.metric(
                                "Confidence",
                                f"{result['confidence']:.2%}",
                                delta=None
                            )
                    
                    with col3:
                        st.metric(
                            "Emoji Score",
                            f"{result['emoji_score']:.3f}",
                            delta=None
                        )
                    
                    # Detailed analysis
                    st.subheader("📋 Detailed Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Original Text:**")
                        st.text(result['text'])
                        
                        st.markdown("**Cleaned Text:**")
                        st.text(result['cleaned_text'])
                    
                    with col2:
                        # Sentiment visualization
                        sentiment_color = get_sentiment_color(result['sentiment'])
                        
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=result['prediction'],
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Sentiment Score"},
                            delta={'reference': 1},
                            gauge={
                                'axis': {'range': [None, 2]},
                                'bar': {'color': sentiment_color},
                                'steps': [
                                    {'range': [0, 0.5], 'color': "#dc3545"},
                                    {'range': [0.5, 1.5], 'color': "#6c757d"},
                                    {'range': [1.5, 2], 'color': "#28a745"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': result['prediction']
                                }
                            }
                        ))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please enter some text to analyze.")
    
    with tab2:
        st.header("Batch Text Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a CSV file with texts (should have a 'text' column)",
            type=['csv']
        )
        
        # Or manual input
        st.markdown("**Or enter multiple texts manually:**")
        manual_texts = st.text_area(
            "Enter texts (one per line):",
            placeholder="I love this! 😍\nThis is terrible 😢\nIt's okay 😐",
            height=200
        )
        
        if st.button("📊 Analyze Batch", type="primary"):
            texts_to_analyze = []
            
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                if 'text' in df.columns:
                    texts_to_analyze = df['text'].astype(str).tolist()
                else:
                    st.error("CSV file must have a 'text' column")
                    return
            elif manual_texts.strip():
                texts_to_analyze = [line.strip() for line in manual_texts.split('\n') if line.strip()]
            else:
                st.warning("Please provide texts to analyze.")
                return
            
            if texts_to_analyze:
                with st.spinner(f"Analyzing {len(texts_to_analyze)} texts..."):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, text in enumerate(texts_to_analyze):
                        result = analyze_text(text, classifier)
                        if result:
                            results.append(result)
                        progress_bar.progress((i + 1) / len(texts_to_analyze))
                
                if results:
                    # Create results DataFrame
                    df_results = pd.DataFrame(results)
                    
                    # Display summary statistics
                    st.subheader("📈 Summary Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Texts", len(results))
                    
                    with col2:
                        positive_count = len(df_results[df_results['sentiment'] == 'Positive'])
                        st.metric("Positive", positive_count)
                    
                    with col3:
                        negative_count = len(df_results[df_results['sentiment'] == 'Negative'])
                        st.metric("Negative", negative_count)
                    
                    with col4:
                        neutral_count = len(df_results[df_results['sentiment'] == 'Neutral'])
                        st.metric("Neutral", neutral_count)
                    
                    # Sentiment distribution chart
                    sentiment_counts = df_results['sentiment'].value_counts()
                    fig = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Sentiment Distribution",
                        color_discrete_map={
                            'Positive': '#28a745',
                            'Negative': '#dc3545',
                            'Neutral': '#6c757d'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Results table
                    st.subheader("📋 Detailed Results")
                    display_df = df_results[['text', 'sentiment', 'confidence', 'emoji_score']].copy()
                    display_df['confidence'] = display_df['confidence'].apply(
                        lambda x: f"{x:.2%}" if x is not None else "N/A"
                    )
                    display_df['emoji_score'] = display_df['emoji_score'].apply(lambda x: f"{x:.3f}")
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Download results
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv"
                    )
    
    with tab3:
        st.header("Model Information")
        
        # Model details
        st.subheader("🔧 Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Model Type:** {classifier.model_type}")
            st.markdown(f"**Emoji Features:** {'Enabled' if classifier.use_emoji_features else 'Disabled'}")
            
            if hasattr(classifier, 'vectorizer') and classifier.vectorizer:
                st.markdown(f"**Vocabulary Size:** {len(classifier.vectorizer.vocabulary_)}")
        
        with col2:
            if hasattr(classifier, 'emoji_lexicon') and classifier.emoji_lexicon:
                st.markdown(f"**Emoji Lexicon Size:** {len(classifier.emoji_lexicon)}")
            
            if hasattr(classifier.model, 'n_estimators'):
                st.markdown(f"**Number of Trees:** {classifier.model.n_estimators}")
        
        # Model performance metrics (placeholder)
        st.subheader("📊 Model Performance")
        st.info("Model performance metrics will be displayed here after training evaluation.")
        
        # Feature importance (if available)
        if hasattr(classifier.model, 'feature_importances_'):
            st.subheader("🎯 Feature Importance")
            # This would show feature importance for tree-based models
            st.info("Feature importance visualization would be displayed here.")

if __name__ == "__main__":
    main()
