import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve

# ==========================================
# 1. Feature Engineering
# ==========================================
def add_medical_features(df):
    df = df.copy()
    df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
    df['pulse_pressure'] = df['bp_systolic'] - df['bp_diastolic']
    return df

def run_svm_optimization():
    print("--- OPTIMIZED SVM (RBF KERNEL) ---")
    print("1. Loading Data...")
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
    except:
        train_df = pd.read_csv('../input/chronicdisease/train.csv')
        test_df = pd.read_csv('../input/chronicdisease/test.csv')

    train_df = add_medical_features(train_df)
    test_df = add_medical_features(test_df)

    X = train_df.drop(columns=['patient_id', 'has_copd_risk'])
    y = train_df['has_copd_risk']
    X_test = test_df.drop(columns=['patient_id'])
    test_ids = test_df['patient_id']

    # ==========================================
    # 2. Preprocessing
    # ==========================================
    # SVMs absolutely REQUIRE StandardScaler to work well
    num_cols = X.select_dtypes(include=['number']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols)
    ])

    # SVM Pipeline
    # kernel='rbf': Handles non-linear data
    # class_weight='balanced': Crucial for F1 score
    # cache_size=1000: Uses 1GB RAM to speed up training
    model = Pipeline([
        ('prep', preprocessor),
        ('clf', SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', cache_size=1000))
    ])

    # ==========================================
    # 3. Threshold Tuning
    # ==========================================
    print("2. Training & Tuning Threshold (This may take 5-10 mins)...")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model.fit(X_train, y_train)
    
    # We use decision_function because predict_proba is extremely slow for SVMs
    val_scores = model.decision_function(X_val)
    
    # Normalize scores to 0-1 range for easier thresholding logic
    val_scores_norm = (val_scores - val_scores.min()) / (val_scores.max() - val_scores.min())
    
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_scores_norm)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    best_idx = np.nanargmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"   Best Validation F1: {best_f1:.4f}")
    print(f"   Optimal Threshold:  {best_thresh:.4f}")

    # ==========================================
    # 4. Final Submission
    # ==========================================
    print("3. Generating Submission...")
    
    # NOTE: For SVM, retraining on full data might take too long. 
    # We will use the model trained on X_train for prediction to be safe.
    
    test_scores = model.decision_function(X_test)
    test_scores_norm = (test_scores - val_scores.min()) / (val_scores.max() - val_scores.min())
    
    test_preds = (test_scores_norm >= best_thresh).astype(int)

    sub = pd.DataFrame({'patient_id': test_ids, 'has_copd_risk': test_preds})
    sub.to_csv('submission_svm_rbf.csv', index=False)
    print("Saved 'submission_svm_rbf.csv'")

if __name__ == "__main__":
    run_svm_optimization()