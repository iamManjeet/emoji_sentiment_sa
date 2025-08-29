#!/usr/bin/env python3
"""
Prediction script for emoji sentiment analysis.
Loads trained models and makes predictions on new text inputs.
"""

import argparse
import sys
import numpy as np
import joblib
from pathlib import Path
from train import EmojiSentimentClassifier
from utils.preprocess import clean_tweet
from utils.emoji_lexicon import emoji_sentiment_feature, load_emoji_sentiment_csv
import re

# Setup paths
BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"

SENTIMENT_LABELS = {
    0: "Negative",
    1: "Positive", 
    2: "Neutral"
}

def predict_sentiment(text, model_path, model_type='logistic', emoji_only: bool = False):
    """
    Predict sentiment for a given text using a trained model.
    
    Args:
        text (str): Input text to analyze
        model_path (str): Path to the trained model
        model_type (str): Type of model ('logistic', 'random_forest', 'transformer')
    
    Returns:
        dict: Prediction results with sentiment label and confidence
    """
    try:
        # Load the trained model directly
        saved_data = joblib.load(model_path)
        
        # Preprocess the text
        cleaned_text = clean_tweet(text)
        
        # Extract text features
        text_features = saved_data['vectorizer'].transform([cleaned_text])
        
        # Extract emoji features if enabled
        emoji_score = 0.0
        if saved_data['use_emoji_features'] and saved_data['emoji_lexicon']:
            emoji_score = emoji_sentiment_feature(text, saved_data['emoji_lexicon'])
            emoji_features = np.array([emoji_score]).reshape(-1, 1)
            
            # Combine features
            from scipy.sparse import hstack
            features = hstack([text_features, emoji_features])
        else:
            features = text_features
        
        # Make prediction (with optional emoji-only override)
        predictions = saved_data['model'].predict(features)
        prediction = predictions[0] if len(predictions) > 0 else 1
        
        # Get sentiment label
        sentiment = SENTIMENT_LABELS.get(prediction, "Unknown")
        
        # For traditional ML models, get prediction probabilities
        confidence = None
        if hasattr(saved_data['model'], 'predict_proba'):
            proba = saved_data['model'].predict_proba(features)[0]

            # Emoji-only mode: rely only on emoji cues when any emoji present
            if emoji_only:
                positive_emojis = {"😊", "😁", "😍", "👍", "🔥", "😎"}
                negative_emojis = {"😡", "😠", "🤬", "😢", "😭", "👎", "🤮", "😞", "😒", "☹️", "😕", "😔", "😩", "😫"}
                has_pos = any(em in text for em in positive_emojis)
                has_neg = any(em in text for em in negative_emojis)
                if has_pos or has_neg:
                    if has_pos and not has_neg:
                        proba = np.array([0.05, 0.95])
                    elif has_neg and not has_pos:
                        proba = np.array([0.95, 0.05])
                    else:
                        # mixed emojis: use aggregate emoji_score sign
                        if emoji_score >= 0:
                            proba = np.array([0.35, 0.65])
                        else:
                            proba = np.array([0.65, 0.35])

            # Heuristic adjustment: if strong negative emoji and clear negative cue, bias towards negative
            cleaned_lower = cleaned_text.lower()
            # Positive cue: negated negative (e.g., "not bad", "not terrible")
            negated_negative_pattern = re.compile(r"\bnot\s+(that\s+|so\s+|very\s+)?(bad|terrible|awful|poor|boring|worse)\b")
            has_negated_negative = bool(negated_negative_pattern.search(cleaned_lower))
            # Angry emoji cue
            angry_emojis = {"😡", "😠", "🤬"}
            has_angry_emoji = any(em in text for em in angry_emojis)
            negative_cues = (
                ' not ', ' no ', "n't", ' never ', ' bad', ' terrible', ' awful', ' worst',
                ' poor', ' hate', ' dislike', ' sucks', ' disappointed', ' boring'
            )
            has_negative_cue = any(cue in f' {cleaned_lower} ' for cue in negative_cues) or 'not good' in cleaned_lower or 'no good' in cleaned_lower
            # If text contains "not bad"-type structure, bias towards positive a bit
            if has_negated_negative:
                pos_p = float(proba[1])
                neg_p = float(proba[0])
                # Ensure positive gets a margin over negative
                desired_pos = max(pos_p, neg_p + 0.25)
                desired_pos = min(0.95, desired_pos)
                desired_neg = 1.0 - desired_pos
                proba = np.array([desired_neg, desired_pos])

            # If angry emoji present and no negated-negative, bias towards negative
            if has_angry_emoji and not has_negated_negative:
                pos_p = float(proba[1])
                neg_p = float(proba[0])
                # Apply a moderate shift toward negative; ensure a small margin
                desired_neg = max(neg_p, pos_p + 0.2)
                desired_neg = min(0.95, desired_neg)
                desired_pos = 1.0 - desired_neg
                proba = np.array([desired_neg, desired_pos])
            if emoji_score <= -0.35 and has_negative_cue:
                # reduce positive probability and increase negative
                pos_p = float(proba[1])
                neg_p = float(proba[0])
                gap = max(0.0, pos_p - neg_p)
                # stronger shift: 40-70% of the gap depending on emoji strength
                shift = min(0.7, max(0.4, abs(emoji_score))) * gap
                pos_p = max(0.0, pos_p - shift)
                neg_p = min(1.0, neg_p + shift)
                # if still close, bias to negative when cues are strong
                if (pos_p - neg_p) < 0.15 and abs(emoji_score) >= 0.5:
                    # nudge slightly further to negative
                    extra = min(0.1, 0.5 * (0.15 - (pos_p - neg_p)))
                    pos_p = max(0.0, pos_p - extra)
                    neg_p = min(1.0, neg_p + extra)
                proba = np.array([neg_p, pos_p])

            # Additional negation pattern: text-only (e.g., "not ... good")
            negation_positive_pattern = re.compile(r"\bnot\s+(a\s+)?(really\s+|very\s+|that\s+|so\s+)?(good|great|amazing|awesome|nice|love|like)\b")
            explicit_negative_pattern = re.compile(r"\b(bad|terrible|awful|poor|worse|worst|hate|dislike|boring|sucks|disappointed)\b")
            has_negated_positive = bool(negation_positive_pattern.search(cleaned_lower))
            has_explicit_negative = bool(explicit_negative_pattern.search(cleaned_lower))
            # If we have a negated-negative (e.g., "not bad"), do NOT apply negative-shifting from this block
            if (has_negated_positive or has_explicit_negative) and not has_negated_negative:
                pos_p = float(proba[1])
                neg_p = float(proba[0])
                gap = max(0.0, pos_p - neg_p)
                base = 0.55 if has_negated_positive else 0.4
                shift = min(0.8, base + 0.2 * (1.0 if has_explicit_negative else 0.0)) * (gap if gap > 0 else 0.25)
                pos_p = max(0.0, pos_p - shift)
                neg_p = min(1.0, neg_p + shift)
                proba = np.array([neg_p, pos_p])

            prediction = int(np.argmax(proba))
            sentiment = SENTIMENT_LABELS.get(prediction, "Unknown")
            confidence = float(np.max(proba))
            
            return {
                'text': text,
                'cleaned_text': cleaned_text,
                'sentiment': sentiment,
                'prediction': prediction,
                'confidence': confidence
            }
        
        return {
            'text': text,
            'cleaned_text': cleaned_text,
            'sentiment': sentiment,
            'prediction': int(prediction),
            'confidence': confidence
        }
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def batch_predict(texts, model_path, model_type='logistic'):
    """
    Predict sentiment for multiple texts.
    
    Args:
        texts (list): List of input texts
        model_path (str): Path to the trained model
        model_type (str): Type of model
    
    Returns:
        list: List of prediction results
    """
    try:
        # Load the trained model
        classifier = EmojiSentimentClassifier.load(model_path)
        
        # Preprocess all texts
        cleaned_texts = [clean_tweet(text) for text in texts]
        
        # Make predictions
        predictions = classifier.predict(cleaned_texts)
        
        # Get confidence scores if available
        confidences = None
        if hasattr(classifier.model, 'predict_proba'):
            probas = classifier.model.predict_proba(cleaned_texts)
            confidences = [max(proba) for proba in probas]
        
        # Format results
        results = []
        for i, (text, cleaned_text, pred) in enumerate(zip(texts, cleaned_texts, predictions)):
            result = {
                'text': text,
                'cleaned_text': cleaned_text,
                'sentiment': SENTIMENT_LABELS.get(pred, "Unknown"),
                'prediction': int(pred),
                'confidence': confidences[i] if confidences else None
            }
            results.append(result)
        
        return results
        
    except Exception as e:
        print(f"Error during batch prediction: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Predict sentiment using trained emoji sentiment analysis model")
    parser.add_argument("--text", type=str, help="Single text to analyze")
    parser.add_argument("--file", type=str, help="File containing texts to analyze (one per line)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--model_type", choices=['logistic', 'random_forest', 'transformer'], 
                       default='logistic', help="Type of model")
    parser.add_argument("--output", type=str, help="Output file for results (JSON format)")
    parser.add_argument("--emoji_only", action='store_true', help="Predict sentiment using emojis only when present")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)
    
    # Single text prediction
    if args.text:
        result = predict_sentiment(args.text, args.model_path, args.model_type, emoji_only=args.emoji_only)
        if result:
            print(f"\nInput Text: {result['text']}")
            print(f"Cleaned Text: {result['cleaned_text']}")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Prediction: {result['prediction']}")
            if result['confidence']:
                print(f"Confidence: {result['confidence']:.4f}")
    
    # Batch prediction from file
    elif args.file:
        if not Path(args.file).exists():
            print(f"Error: Input file not found at {args.file}")
            sys.exit(1)
        
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"Analyzing {len(texts)} texts...")
        results = batch_predict(texts, args.model_path, args.model_type)
        
        if results:
            # Print results
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Text: {result['text']}")
                print(f"   Sentiment: {result['sentiment']}")
                if result['confidence']:
                    print(f"   Confidence: {result['confidence']:.4f}")
            
            # Save to output file if specified
            if args.output:
                import json
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"\nResults saved to: {args.output}")
    
    else:
        print("Please provide either --text or --file argument")
        parser.print_help()

if __name__ == "__main__":
    main()
