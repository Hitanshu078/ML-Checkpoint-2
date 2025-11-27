#*****
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

def train_best_sklearn_nn():
    X, y, X_test = preprocess_data()

    # Train-val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training strongest sklearn NN...")
    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-4,              # L2 regularization
        batch_size=256,
        learning_rate='adaptive',
        learning_rate_init=0.0008,
        max_iter=40,             # keep small (large dataset)
        early_stopping=False,
        random_state=42,
        verbose=True
    )

    model.fit(X_train, y_train)

    # Threshold tuning
    print("Finding best F1 threshold...")
    val_probs = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    f1s = 2*(precisions*recalls)/(precisions+recalls + 1e-9)

    best_idx = np.argmax(f1s)
    best_th = thresholds[best_idx]
    best_f1 = f1s[best_idx]

    print(f"BEST F1 = {best_f1:.4f}")
    print(f"BEST THRESHOLD = {best_th:.4f}")

    # Predict test
    test_probs = model.predict_proba(X_test)[:, 1]
    preds = (test_probs >= best_th).astype(int)

        # Submission (correct version)
    test_df = pd.read_csv('/kaggle/input/chronicdisease/test.csv')

    submission = pd.DataFrame({
        'patient_id': test_df['patient_id'],
        'has_copd_risk': preds
    })

    submission.to_csv('best_nn_submission.csv', index=False)
    print("Saved best_nn_submission.csv")

