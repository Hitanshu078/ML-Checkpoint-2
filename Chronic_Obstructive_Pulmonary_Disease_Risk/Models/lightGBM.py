# gave score of 0.744
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

def train_lightgbm():
    print("Loading data...")
    X = pd.read_csv('processed_X_train.csv')
    y = pd.read_csv('processed_y_train.csv').values.ravel()
    X_test = pd.read_csv('processed_X_test.csv')
    ids = pd.read_csv('test_ids.csv')

    # Class imbalance fix
    pos = y.sum()
    neg = len(y) - pos
    scale_weight = neg / pos
    print(f"scale_pos_weight = {scale_weight:.2f}")

    # Train/Val Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model
    print("Training LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=40,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Threshold tuning
    print("Optimizing threshold...")
    val_probs = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

    idx = np.argmax(f1s)
    best_th = thresholds[idx]
    best_f1 = f1s[idx]

    print(f"Best F1: {best_f1:.4f} at threshold {best_th:.4f}")

    # Predict on Test
    test_probs = model.predict_proba(X_test)[:, 1]
    preds = (test_probs >= best_th).astype(int)

    # Submission
    submission = pd.DataFrame({
        'patient_id': ids['patient_id'],
        'has_copd_risk': preds
    })

    submission.to_csv('lightgbm_submission.csv', index=False)
    print("Saved lightgbm_submission.csv")

if __name__ == "__main__":
    train_lightgbm()
