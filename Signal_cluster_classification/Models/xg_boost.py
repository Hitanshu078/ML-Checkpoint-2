# ============================================================
# XGBOOST BEST MODEL
# ============================================================


import pandas as pd
import numpy as np
from xgboost import XGBClassifier
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
    'max_depth': trial.suggest_int('max_depth', 2, 8),
    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    'gamma': trial.suggest_float('gamma', 0, 5),
    'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
    'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'tree_method': 'hist',
    'random_state': 42,
    'use_label_encoder': False
  }
  X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
    X_train, y_train, weights, test_size=0.2, stratify=y_train, random_state=42
  )
  model = XGBClassifier(**param)
  model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)])
  val_pred = np.argmax(model.predict_proba(X_val), axis=1)
  score = f1_score(y_val, val_pred, average='macro')
  return 1.0 - score  # minimize 1-F1

print("Starting Optuna hyperparameter search for XGBoost...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=40, show_progress_bar=True)

print("Best params:", study.best_params)

# Retrain on full data with best params
best_params = study.best_params.copy()
best_params.update({
  'objective': 'multi:softprob',
  'num_class': 3,
  'eval_metric': 'mlogloss',
  'tree_method': 'hist',
  'random_state': 42,
  'use_label_encoder': False
})

X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
  X_train, y_train, weights, test_size=0.2, stratify=y_train, random_state=42
)
model = XGBClassifier(**best_params)
model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)])
val_pred = np.argmax(model.predict_proba(X_val), axis=1)
val_f1 = f1_score(y_val, val_pred, average='macro')
print(f"Validation Macro F1 with best params: {val_f1:.5f}")

# Retrain on all data for final prediction
final_model = XGBClassifier(**best_params)
final_model.fit(X_train, y_train, sample_weight=weights)
pred = np.argmax(final_model.predict_proba(X_test), axis=1)
pred = label_classes[pred]

# Save submission
pd.DataFrame({"sample_id": test_ids["sample_id"], "category": pred}) \
  .to_csv("submission_xgb.csv", index=False)

print("Saved submission_xgb.csv with tuned XGBoost model.")
print("Best XGBoost parameters:")
for k, v in study.best_params.items():
  print(f"  {k}: {v}")
print(f"Validation Macro F1: {val_f1:.5f}")
