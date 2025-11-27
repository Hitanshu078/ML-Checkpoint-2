import pandas as pd
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# ================= CONFIGURATION =================
# New preprocessed data paths
X_train_path = 'preprocessed_data/X_train_final.csv'
y_train_path = 'preprocessed_data/y_final.csv'
X_test_path = 'preprocessed_data/X_test_final.csv'
test_ids_path = 'preprocessed_data/test_ids.csv'
# =================================================


print(f"Loading test IDs from: {test_ids_path}")
test_ids_df = pd.read_csv(test_ids_path)

# Load features and target separately, then merge
X = pd.read_csv(X_train_path)
y = pd.read_csv(y_train_path)['target']
X_test_final = pd.read_csv(X_test_path)




# 3. CALCULATE F1 MACRO SCORE (Validation Step) with StandardScaler and class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

print("\n--- Step A: Validating Model ---")
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

mlp_val = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', max_iter=1000, random_state=42)
)
mlp_val.fit(X_train_split, y_train_split)
y_val_pred = mlp_val.predict(X_val_split)

val_f1 = f1_score(y_val_split, y_val_pred, average='macro')
print(f"Validation F1 Macro Score: {val_f1:.5f}")


# 4. GENERATE SUBMISSION (Final Step)
print("\n--- Step B: Generating Submission File ---")
mlp_final = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', max_iter=1000, random_state=42)
)

print("Retraining on full dataset...")
mlp_final.fit(X, y)

print("Predicting on test set...")
predictions_numeric = mlp_final.predict(X_test_final)

# Convert to labels
reverse_mapping = {0: 'Group_A', 1: 'Group_B', 2: 'Group_C'}
predictions_labels = [reverse_mapping[pred] for pred in predictions_numeric]

# Ensure we only take the first N rows if IDs match test data length (safety check)
if len(test_ids_df) != len(predictions_labels):
    print(f"Warning: ID count ({len(test_ids_df)}) matches Prediction count ({len(predictions_labels)})?")
    # Usually exact match is required.

submission = pd.DataFrame({
    'sample_id': test_ids_df['sample_id'],
    'category': predictions_labels
})

submission.to_csv('NNsubmission.csv', index=False)
print("Success! 'NNsubmission.csv' created successfully.")