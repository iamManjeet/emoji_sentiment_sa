#!/usr/bin/env python3
"""
Main training script for emoji sentiment analysis.
Supports multiple model types and emoji feature integration.
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch
from tqdm import tqdm

from utils.preprocess import clean_tweet
from utils.emoji_lexicon import emoji_sentiment_feature, load_emoji_sentiment_csv
from utils.metrics import compute_metrics

# Setup paths
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
PROC = DATA / "processed"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

class EmojiSentimentClassifier:
    """Classifier that combines text features with emoji sentiment features."""
    
    def __init__(self, model_type='logistic', use_emoji_features=True, emoji_lexicon_path=None):
        self.model_type = model_type
        self.use_emoji_features = use_emoji_features
        self.emoji_lexicon = load_emoji_sentiment_csv(emoji_lexicon_path) if use_emoji_features else None
        self.model = None
        self.vectorizer = None
        
    def extract_features(self, texts):
        """Extract text and emoji features from input texts."""
        # Text features using TF-IDF
        if self.vectorizer is None:
            # Retain negation words by removing them from the stopword list
            negation_words = {"not", "no", "never", "nor"}
            custom_stop_words = sorted(list(ENGLISH_STOP_WORDS.difference(negation_words)))
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words=custom_stop_words,
                min_df=5,
                token_pattern=r"(?u)\b\w[\w']+\b"  # keep tokens like n't
            )
            text_features = self.vectorizer.fit_transform(texts)
        else:
            text_features = self.vectorizer.transform(texts)
        
        if not self.use_emoji_features:
            return text_features
        
        # Emoji sentiment features
        emoji_features = np.array([
            emoji_sentiment_feature(text, self.emoji_lexicon) 
            for text in texts
        ]).reshape(-1, 1)
        
        # Combine features
        from scipy.sparse import hstack
        combined_features = hstack([text_features, emoji_features])
        return combined_features
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the sentiment classifier."""
        print(f"Training {self.model_type} model...")
        
        if self.model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=3000,
                C=0.8,
                class_weight='balanced',
                n_jobs=-1,
                solver='lbfgs'
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'transformer':
            self._train_transformer(X_train, y_train, X_val, y_val)
            return
        
        # Extract features
        X_train_features = self.extract_features(X_train)
        
        # Train model
        self.model.fit(X_train_features, y_train)
        
        # Validation if provided
        if X_val is not None and y_val is not None:
            X_val_features = self.extract_features(X_val)
            y_pred = self.model.predict(X_val_features)
            metrics = compute_metrics(y_val, y_pred)
            print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
            print(f"Validation Macro F1: {metrics['macro_f1']:.4f}")
    
    def _train_transformer(self, X_train, y_train, X_val, y_val):
        """Train a transformer-based model."""
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=3
        )
        
        # Prepare datasets
        train_dataset = Dataset.from_dict({
            'text': X_train,
            'label': y_train
        })
        val_dataset = Dataset.from_dict({
            'text': X_val,
            'label': y_val
        })
        
        def tokenize_function(examples):
            return tokenizer(
                examples['text'], 
                padding='max_length', 
                truncation=True, 
                max_length=128
            )
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(MODELS / "transformer"),
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=str(MODELS / "transformer" / "logs"),
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        trainer.train()
        self.model = model
        self.tokenizer = tokenizer
    
    def predict(self, texts):
        """Predict sentiment for input texts."""
        if self.model_type == 'transformer':
            return self._predict_transformer(texts)
        
        features = self.extract_features(texts)
        return self.model.predict(features)
    
    def _predict_transformer(self, texts):
        """Predict using transformer model."""
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        return predictions.cpu().numpy()
    
    def save(self, filepath):
        """Save the trained model."""
        if self.model_type == 'transformer':
            self.model.save_pretrained(filepath)
            self.tokenizer.save_pretrained(filepath)
        else:
            joblib.dump({
                'model': self.model,
                'vectorizer': self.vectorizer,
                'emoji_lexicon': self.emoji_lexicon,
                'use_emoji_features': self.use_emoji_features,
                'model_type': self.model_type
            }, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load a trained model."""
        if Path(filepath).is_dir():
            # Transformer model
            model = AutoModelForSequenceClassification.from_pretrained(filepath)
            tokenizer = AutoTokenizer.from_pretrained(filepath)
            instance = cls(model_type='transformer')
            instance.model = model
            instance.tokenizer = tokenizer
        else:
            # Traditional ML model
            saved_data = joblib.load(filepath)
            instance = cls(
                model_type=saved_data['model_type'],
                use_emoji_features=saved_data['use_emoji_features']
            )
            instance.model = saved_data['model']
            instance.vectorizer = saved_data['vectorizer']
            instance.emoji_lexicon = saved_data['emoji_lexicon']
        
        return instance

def load_data():
    """Load processed datasets."""
    train_df = pd.read_csv(PROC / "train.csv")
    val_df = pd.read_csv(PROC / "val.csv")
    test_df = pd.read_csv(PROC / "test.csv")
    
    return (
        train_df['text_clean'].tolist(),
        train_df['label'].tolist(),
        val_df['text_clean'].tolist(),
        val_df['label'].tolist(),
        test_df['text_clean'].tolist(),
        test_df['label'].tolist()
    )

def main():
    parser = argparse.ArgumentParser(description="Train emoji sentiment analysis model")
    parser.add_argument("--model_type", choices=['logistic', 'random_forest', 'transformer'], 
                       default='logistic', help="Type of model to train")
    parser.add_argument("--use_emoji_features", action='store_true', 
                       help="Whether to use emoji sentiment features")
    parser.add_argument("--emoji_lexicon_path", type=str, 
                       help="Path to emoji sentiment lexicon CSV")
    parser.add_argument("--output_dir", type=str, default=str(MODELS),
                       help="Directory to save trained model")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Train model
    classifier = EmojiSentimentClassifier(
        model_type=args.model_type,
        use_emoji_features=args.use_emoji_features,
        emoji_lexicon_path=args.emoji_lexicon_path
    )
    
    classifier.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    y_pred = classifier.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    
    print("\nTest Set Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("\nDetailed Classification Report:")
    print(metrics['report'])
    
    # Save model
    output_path = Path(args.output_dir) / f"{args.model_type}_emoji_{args.use_emoji_features}.joblib"
    if args.model_type == 'transformer':
        output_path = Path(args.output_dir) / f"{args.model_type}_emoji_{args.use_emoji_features}"
    
    classifier.save(str(output_path))
    print(f"\nModel saved to: {output_path}")

if __name__ == "__main__":
    main()
