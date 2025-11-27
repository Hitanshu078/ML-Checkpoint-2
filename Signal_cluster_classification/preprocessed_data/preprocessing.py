# ============================================================
# UNIVERSAL PREPROCESSING PIPELINE + SAVE AS CSV
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler, LabelEncoder

# ------------------------
# LOAD DATA
# ------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

test_ids = test["sample_id"]
train = train.drop(columns=["sample_id"])
test = test.drop(columns=["sample_id"])

# ------------------------
# POWER TRANSFORM
# ------------------------
pt = PowerTransformer()
train[["signal_strength", "response_level"]] = pt.fit_transform(
    train[["signal_strength", "response_level"]]
)
test[["signal_strength", "response_level"]] = pt.transform(
    test[["signal_strength", "response_level"]]
)

# ------------------------
# STANDARD SCALE
# ------------------------
scaler = StandardScaler()
train[["signal_strength", "response_level"]] = scaler.fit_transform(
    train[["signal_strength", "response_level"]]
)
test[["signal_strength", "response_level"]] = scaler.transform(
    test[["signal_strength", "response_level"]]
)

# ------------------------
# LABEL ENCODE TARGET
# ------------------------
le = LabelEncoder()
train["target"] = le.fit_transform(train["category"])

# Save encoder classes (optional)
pd.Series(le.classes_).to_csv("label_classes.csv", index=False)

# ------------------------
# FEATURE ENGINEERING
# ------------------------
def add_features(df):
    df["interaction"] = df["signal_strength"] * df["response_level"]
    df["magnitude"] = np.sqrt(df["signal_strength"]**2 + df["response_level"]**2)
    return df

train = add_features(train)
test = add_features(test)

# ------------------------
# FINAL FEATURES
# ------------------------
features = ["signal_strength", "response_level", "interaction", "magnitude"]

X_train_final = train[features]
y_final = train["target"]
X_test_final = test[features]

# ------------------------
# SAVE OUTPUTS
# ------------------------

X_train_final.to_csv("X_train_final.csv", index=False)
X_test_final.to_csv("X_test_final.csv", index=False)
y_final.to_csv("y_final.csv", index=False)

# save test_ids
test_ids.to_csv("test_ids.csv", index=False)

print("Preprocessing Completed!")
print("Saved: X_train_final.csv, X_test_final.csv, y_final.csv, test_ids.csv")
print("Shapes:", X_train_final.shape, X_test_final.shape)
