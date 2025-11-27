import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, precision_recall_curve

def run_pure_optimization():
    print("1. Loading Data...")
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
    except:
        train_df = pd.read_csv('../input/chronicdisease/train.csv')
        test_df = pd.read_csv('../input/chronicdisease/test.csv')

    # Strictly using given columns only
    X = train_df.drop(columns=['patient_id', 'has_copd_risk'])
    y = train_df['has_copd_risk']
    
    X_test = test_df.drop(columns=['patient_id'])
    test_ids = test_df['patient_id']

    # ==========================================
    # 2. Advanced Preprocessing (The Key Upgrade)
    # ==========================================
    num_cols = X.select_dtypes(include=['number']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    # Pipeline for Numerical Data
    # 1. Fill missing (Median)
    # 2. Yeo-Johnson Transform: Forces data into a Bell Curve (Crucial for LogReg)
    # 3. Scale: Standardize to mean=0, std=1
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('power', PowerTransformer(method='yeo-johnson')), 
        ('scaler', StandardScaler())
    ])

    # Pipeline for Categorical Data
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    # ==========================================
    # 3. Model & Hyperparameter Tuning
    # ==========================================
    # Solver 'saga' is required for ElasticNet
    # Class Weight 'balanced' handles the unequal positive/negative cases
    model = Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(
            solver='saga', 
            class_weight='balanced', 
            max_iter=5000, 
            random_state=42
        ))
    ])

    # Grid Search to find the perfect math settings
    # l1_ratio: 0=Ridge, 1=Lasso, 0.5=Mix (ElasticNet)
    param_grid = {
        'clf__C': [0.01, 0.1, 0.5, 1, 5, 10],
        'clf__penalty': ['elasticnet'],
        'clf__l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]
    }

    print("2. Tuning Hyperparameters (Grid Search)...")
    # Using Stratified Cross-Validation to ensure reliability
    grid = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    
    # Reserve a validation set for final threshold tuning
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    print(f"   Best Params: {grid.best_params_}")

    # ==========================================
    # 4. Threshold Optimization
    # ==========================================
    print("3. Optimizing Threshold...")
    # Get probability scores
    val_probs = best_model.predict_proba(X_val)[:, 1]
    
    # Find the exact cutoff that maximizes F1
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    
    best_idx = np.nanargmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"   Best Validation F1: {best_f1:.4f}")
    print(f"   Optimal Threshold:  {best_thresh:.4f}")

    # ==========================================
    # 5. Final Prediction
    # ==========================================
    print("4. Generating Submission...")
    # Refit on ALL data to capture every bit of information
    best_model.fit(X, y)
    
    test_probs = best_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_thresh).astype(int)

    submission = pd.DataFrame({
        'patient_id': test_ids,
        'has_copd_risk': test_preds
    })
    
    submission.to_csv('submission_pure_logreg.csv', index=False)
    print("Success! Saved 'submission_pure_logreg.csv'")

if __name__ == "__main__":
    run_pure_optimization()