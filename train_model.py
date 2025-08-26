import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load dataset
df = pd.read_csv(r"C:\Users\drbdp\Downloads\archive (2)\creditcard.csv")

# Features: drop 'Class' (fraud label), keep rest
X = df.drop(columns=["Class"])
y = df["Class"]

# Train/test split (stratify keeps fraud ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. Train Random Forest (simple, good for imbalance)
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# 3. Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

# 4. Save model
joblib.dump(model, "fraud_model.pkl")
print("✅ Model saved as fraud_model.pkl")
