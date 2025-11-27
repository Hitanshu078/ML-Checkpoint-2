# ============================================================
# RANDOM FOREST BEST MODEL WITH HYPERPARAMETER TUNING
# ============================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score
import optuna

# Load data
X_train = pd.read_csv("preprocessed_data/X_train_final.csv")
X_test = pd.read_csv("preprocessed_data/X_test_final.csv")
y_train = pd.read_csv("preprocessed_data/y_final.csv").values.ravel()
test_ids = pd.read_csv("preprocessed_data/test_ids.csv")
label_classes = pd.read_csv("preprocessed_data/label_classes.csv").values.ravel()

# Class balancing
weights = compute_sample_weight("balanced", y_train)

# Hyperparameter tuning with Optuna
def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_train, y_train, weights, test_size=0.2, stratify=y_train, random_state=42
    )
    model = RandomForestClassifier(**param)
    model.fit(X_tr, y_tr, sample_weight=w_tr)
    val_pred = model.predict(X_val)
    score = f1_score(y_val, val_pred, average='macro')
    return 1.0 - score  # minimize 1-F1

print("Starting Optuna hyperparameter search for Random Forest...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=40, show_progress_bar=True)

print("Best params:", study.best_params)

# Retrain on full data with best params
best_params = study.best_params.copy()
best_params.update({
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
})

X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
    X_train, y_train, weights, test_size=0.2, stratify=y_train, random_state=42
)
model = RandomForestClassifier(**best_params)
model.fit(X_tr, y_tr, sample_weight=w_tr)
val_pred = model.predict(X_val)
val_f1 = f1_score(y_val, val_pred, average='macro')
print(f"Validation Macro F1 with best params: {val_f1:.5f}")

# Retrain on all data for final prediction
final_model = RandomForestClassifier(**best_params)
final_model.fit(X_train, y_train, sample_weight=weights)
pred = final_model.predict(X_test)
pred = label_classes[pred]

# Save submission
pd.DataFrame({"sample_id": test_ids["sample_id"], "category": pred}) \
  .to_csv("submission_rf.csv", index=False)

print("Saved submission_rf.csv with tuned Random Forest model.")
print("Best Random Forest parameters:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
print(f"Validation Macro F1: {val_f1:.5f}")
