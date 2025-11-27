# ============================================================
# LIGHTGBM BEST MODEL
# ============================================================


import pandas as pd
import numpy as np
import lightgbm as lgb
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
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 2.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 2.0),
        'random_state': 42,
        'verbosity': -1
    }
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_train, y_train, weights, test_size=0.2, stratify=y_train, random_state=42
    )
    train_data = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
    val_data = lgb.Dataset(X_val, label=y_val, weight=w_val)
    model = lgb.train(
        param,
        train_data,
        valid_sets=[val_data],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(100)]
    )
    val_pred = np.argmax(model.predict(X_val), axis=1)
    score = f1_score(y_val, val_pred, average='macro')
    return 1.0 - score  # minimize 1-F1

print("Starting Optuna hyperparameter search for LightGBM...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=40, show_progress_bar=True)

print("Best params:", study.best_params)

# Retrain on full data with best params
best_params = study.best_params.copy()
best_params.update({
    'objective': 'multiclass',
    'num_class': 3,
    'random_state': 42,
    'verbosity': -1
})

X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
    X_train, y_train, weights, test_size=0.2, stratify=y_train, random_state=42
)
train_data = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
val_data = lgb.Dataset(X_val, label=y_val, weight=w_val)
model = lgb.train(
    best_params,
    train_data,
    valid_sets=[val_data],
    num_boost_round=2000,
    callbacks=[lgb.early_stopping(100)]
)
val_pred = np.argmax(model.predict(X_val), axis=1)
val_f1 = f1_score(y_val, val_pred, average='macro')
print(f"Validation Macro F1 with best params: {val_f1:.5f}")

# Retrain on all data for final prediction
final_data = lgb.Dataset(X_train, label=y_train, weight=weights)
final_model = lgb.train(
    best_params,
    final_data,
    num_boost_round=model.best_iteration or 2000
)
pred = np.argmax(final_model.predict(X_test), axis=1)
pred = label_classes[pred]

# Save submission
pd.DataFrame({"sample_id": test_ids["sample_id"], "category": pred}) \
  .to_csv("submission_lgbm.csv", index=False)

print("Saved submission_lgbm.csv with tuned LightGBM model.")
print("Best LightGBM parameters:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
print(f"Validation Macro F1: {val_f1:.5f}")
