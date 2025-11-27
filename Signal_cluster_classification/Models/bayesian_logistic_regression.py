# ============================================================
# BAYESIAN LOGISTIC REGRESSION (L2 PRIOR)
# ============================================================


import pandas as pd
import numpy as np
from skopt import BayesSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load data
X_train = pd.read_csv("preprocessed_data/X_train_final.csv")
X_test = pd.read_csv("preprocessed_data/X_test_final.csv")
y_train = pd.read_csv("preprocessed_data/y_final.csv").values.ravel()
test_ids = pd.read_csv("preprocessed_data/test_ids.csv")
label_classes = pd.read_csv("preprocessed_data/label_classes.csv").values.ravel()

# Class balancing
weights = compute_sample_weight("balanced", y_train)

# Validation split for reporting
X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
  X_train, y_train, weights, test_size=0.2, stratify=y_train, random_state=42
)

# Pipeline for scaling + logistic regression
pipe = make_pipeline(
  StandardScaler(),
  LogisticRegression(multi_class='multinomial', random_state=42, max_iter=3000)
)

# Bayesian search space
search_spaces = {
  'logisticregression__C': (1e-4, 1e3, 'log-uniform'),
  'logisticregression__solver': ['lbfgs', 'saga'],
  'logisticregression__penalty': ['l2', None],
  'logisticregression__tol': (1e-6, 1e-2, 'log-uniform'),
}

# Bayesian optimization
bayes_cv = BayesSearchCV(
  pipe,
  search_spaces,
  n_iter=60,
  scoring='f1_macro',
  cv=StratifiedKFold(n_splits=7, shuffle=True, random_state=42),
  n_jobs=-1,
  verbose=2,
  refit=True,
  random_state=42
)
bayes_cv.fit(X_tr, y_tr, logisticregression__sample_weight=w_tr)

print("Best params:", bayes_cv.best_params_)

# Validation F1
val_pred = bayes_cv.predict(X_val)
val_f1 = f1_score(y_val, val_pred, average='macro')
print(f"Validation Macro F1 with best params: {val_f1:.5f}")

# Retrain on all data for final prediction

# Prepare final model params, handle penalty=None
final_params = dict(
  multi_class='multinomial',
  random_state=42,
  max_iter=3000,
  C=bayes_cv.best_params_['logisticregression__C'],
  solver=bayes_cv.best_params_['logisticregression__solver'],
  tol=bayes_cv.best_params_['logisticregression__tol']
)
if bayes_cv.best_params_['logisticregression__penalty'] is not None:
  final_params['penalty'] = bayes_cv.best_params_['logisticregression__penalty']

final_pipe = make_pipeline(
  StandardScaler(),
  LogisticRegression(**final_params)
)
final_pipe.fit(X_train, y_train, logisticregression__sample_weight=weights)
pred = final_pipe.predict(X_test)
pred = label_classes[pred]

# Save submission
pd.DataFrame({"sample_id": test_ids["sample_id"], "category": pred}) \
  .to_csv("submission_bayesian_logreg.csv", index=False)

print("Saved submission_bayesian_logreg.csv with tuned Bayesian Logistic Regression model.")
print("Best Bayesian Logistic Regression parameters:")
for k, v in bayes_cv.best_params_.items():
  print(f"  {k}: {v}")
print(f"Validation Macro F1: {val_f1:.5f}")
