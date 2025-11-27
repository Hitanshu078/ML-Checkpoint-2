import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_curve

def train_xgboost():
    # 1. Load the Preprocessed Data
    print("Loading data...")
    try:
        X = pd.read_csv('processed_X_train.csv')
        y = pd.read_csv('processed_y_train.csv')
        X_test = pd.read_csv('processed_X_test.csv')
        ids = pd.read_csv('test_ids.csv')
    except FileNotFoundError:
        print("Error: Processed files not found. Run the preprocessing script first.")
        return

    # Ensure y is a 1D array
    y = y.values.ravel()

    # 2. Calculate Scale Weight for Imbalance
    # Formula: (Number of Negatives) / (Number of Positives)
    # This tells the model: "Pay X times more attention to the positive class"
    pos_count = y.sum()
    neg_count = len(y) - pos_count
    scale_weight = neg_count / pos_count
    print(f"Class Imbalance Detected. Using scale_pos_weight: {scale_weight:.2f}")

    # 3. Split Data for Validation
    # We use this to find the best F1 threshold without cheating
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Initialize and Train XGBoost
    print("Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=1000,          # High number, but we use early_stopping
        learning_rate=0.05,         # Slow learning for better accuracy
        max_depth=6,                # Standard depth for tabular data
        subsample=0.8,              # Prevent overfitting
        colsample_bytree=0.8,       # Prevent overfitting
        scale_pos_weight=scale_weight, # CRITICAL: Fixes low Recall/F1
        random_state=42,
        n_jobs=-1                   # Use all CPU cores
    )

    # Train with Early Stopping
    # Stop training if the validation score doesn't improve for 50 rounds
    model.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)], 
        eval_metric='logloss',
        early_stopping_rounds=50,
        verbose=100
    )

    # 5. Threshold Tuning (The "Secret Sauce")
    print("\nOptimizing Decision Threshold...")
    
    # Predict probabilities (e.g., 0.85, 0.12) instead of labels (0, 1)
    val_probs = model.predict_proba(X_val)[:, 1]
    
    # Calculate F1 for every possible threshold
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    
    # Find the single best threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"Best Validation F1 Score: {best_f1:.4f}")
    print(f"Optimal Threshold: {best_threshold:.4f}")

    # 6. Generate Submission
    print("Generating predictions for Test set...")
    test_probs = model.predict_proba(X_test)[:, 1]
    
    # Apply the OPTIMAL threshold (not default 0.5)
    test_preds = (test_probs >= best_threshold).astype(int)

    submission = pd.DataFrame({
        'patient_id': ids['patient_id'],
        'has_copd_risk': test_preds
    })
    
    submission.to_csv('xgboost_submission.csv', index=False)
    print("Success! Saved 'xgboost_submission.csv'")

if __name__ == "__main__":
    train_xgboost()