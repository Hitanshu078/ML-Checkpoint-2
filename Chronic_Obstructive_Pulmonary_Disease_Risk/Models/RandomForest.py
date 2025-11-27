# 0.771
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

def train_random_forest():
    print("Loading data...")
    X = pd.read_csv('/kaggle/input/chronicdisease/processed_X_train.csv')
    y = pd.read_csv('/kaggle/input/chronicdisease/processed_y_train.csv').values.ravel()
    X_test = pd.read_csv('/kaggle/input/chronicdisease/processed_X_test.csv')
    ids = pd.read_csv('/kaggle/input/chronicdisease/test_ids.csv')

    # Class imbalance
    pos = y.sum()
    neg = len(y) - pos
    scale_weight = neg / pos
    print(f"Class weight multiplier = {scale_weight:.2f}")

    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model
    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=700,
        max_depth=None,
        class_weight={0:1, 1:scale_weight},
        random_state=42,
        n_jobs=-1
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

    # Predict
    test_probs = model.predict_proba(X_test)[:, 1]
    preds = (test_probs >= best_th).astype(int)

    submission = pd.DataFrame({
        'patient_id': ids['patient_id'],
        'has_copd_risk': preds
    })

    submission.to_csv('rf_submission.csv', index=False)
    print("Saved rf_submission.csv")

if __name__ == "__main__":
    train_random_forest()
