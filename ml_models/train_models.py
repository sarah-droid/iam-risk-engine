import pandas as pd
import numpy as np
import joblib
import os
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import shap

# ── Load features ─────────────────────────────────────────────────────────────
df = pd.read_csv("datasets/features.csv")
print(f"Loaded {len(df)} rows from features.csv")
print(f"Label distribution:\n{df['risk_label'].value_counts().to_string()}\n")

# ── Define ML feature columns (no IDs or labels) ─────────────────────────────
FEATURE_COLS = [
    "impossible_travel",
    "hours_since_last",
    "mfa_fail_count",
    "is_known_device",
    "country_deviation",
    "hour_sin",
    "hour_cos",
    "is_business_hours",
    "role_rank",
    "role_escalation",
    "pwd_reset_7d",
    "peer_zscore",
]

X = df[FEATURE_COLS].copy()
y_raw = df["risk_label"].copy()

# ── Encode labels: LOW=0, MEDIUM=1, HIGH=2 ───────────────────────────────────
label_map     = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
label_map_inv = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
y = y_raw.map(label_map)

print(f"Feature matrix shape : {X.shape}")
print(f"Features used        : {FEATURE_COLS}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — Isolation Forest (unsupervised anomaly detection)
# Finds unusual events without needing labels
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 55)
print("Training Isolation Forest ...")
print("=" * 55)

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.08,   # ~8% of data expected to be anomalous
    max_samples="auto",
    random_state=42,
)
iso_forest.fit(X)

# decision_function: more negative = more anomalous
iso_scores_raw = iso_forest.decision_function(X)

# Normalise to 0–100 (higher = more anomalous / riskier)
iso_min = iso_scores_raw.min()
iso_max = iso_scores_raw.max()
iso_scores_normalised = 100 * (1 - (iso_scores_raw - iso_min) / (iso_max - iso_min))

df["iso_score"] = np.round(iso_scores_normalised, 2)
print(f"Isolation Forest scores — min: {df['iso_score'].min():.1f}  max: {df['iso_score'].max():.1f}  mean: {df['iso_score'].mean():.1f}")
print(f"Anomaly predictions    — normal: {(iso_forest.predict(X)==1).sum()}  anomaly: {(iso_forest.predict(X)==-1).sum()}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — XGBoost Classifier (supervised risk classification)
# Classifies each event as LOW / MEDIUM / HIGH
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 55)
print("Training XGBoost Classifier ...")
print("=" * 55)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=["LOW", "MEDIUM", "HIGH"]))

# XGBoost confidence probabilities
xgb_proba = xgb_model.predict_proba(X)   # shape: (n, 3)
df["xgb_low_prob"]    = np.round(xgb_proba[:, 0], 4)
df["xgb_medium_prob"] = np.round(xgb_proba[:, 1], 4)
df["xgb_high_prob"]   = np.round(xgb_proba[:, 2], 4)
df["xgb_pred_label"]  = xgb_model.predict(X)
df["xgb_pred_label"]  = df["xgb_pred_label"].map(label_map_inv)

# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED RISK SCORE
# Blend IsolationForest (40%) + XGBoost HIGH probability (60%)
# Result: 0–100 integer score
# ═══════════════════════════════════════════════════════════════════════════════
df["final_risk_score"] = np.round(
    0.40 * df["iso_score"] +
    0.60 * (df["xgb_high_prob"] * 100)
).astype(int)

df["final_risk_score"] = df["final_risk_score"].clip(0, 100)

print("\nFinal Risk Score stats:")
print(df.groupby("risk_label")["final_risk_score"].describe()[["mean","min","max"]].to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# SHAP — Explainability
# Tells us WHY each score is high (used in dashboard explanation panel)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("Computing SHAP values for explainability ...")
print("=" * 55)

explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X)   # shape: (n, features, classes)

# For each row, pick the SHAP values for the predicted class
# and find the top 3 most influential features
shap_explanations = []
for i in range(len(X)):
    pred_class = int(xgb_model.predict(X.iloc[[i]])[0])
    if isinstance(shap_values, list):
        row_shap = shap_values[pred_class][i]
    else:
        row_shap = shap_values[i, :, pred_class]

    # Sort features by absolute SHAP value descending
    feat_importance = sorted(
        zip(FEATURE_COLS, row_shap),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    top3 = [
        {"feature": f, "shap": round(float(s), 4)}
        for f, s in feat_importance[:3]
    ]
    shap_explanations.append(json.dumps(top3))

df["shap_explanation"] = shap_explanations

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE EVERYTHING
# ═══════════════════════════════════════════════════════════════════════════════
os.makedirs("ml_models", exist_ok=True)
os.makedirs("datasets",  exist_ok=True)

# Save trained models
joblib.dump(iso_forest, "ml_models/isolation_forest.pkl")
joblib.dump(xgb_model,  "ml_models/xgboost_model.pkl")
joblib.dump(explainer,  "ml_models/shap_explainer.pkl")

# Save scoring metadata
meta = {
    "feature_cols":   FEATURE_COLS,
    "label_map":      label_map,
    "label_map_inv":  label_map_inv,
    "iso_min":        float(iso_min),
    "iso_max":        float(iso_max),
    "iso_weight":     0.40,
    "xgb_weight":     0.60,
}
with open("ml_models/model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

# Save scored dataset
df.to_csv("datasets/scored_events.csv", index=False)

print("\n✅ Saved:")
print("   ml_models/isolation_forest.pkl")
print("   ml_models/xgboost_model.pkl")
print("   ml_models/shap_explainer.pkl")
print("   ml_models/model_meta.json")
print("   datasets/scored_events.csv")

# ── Sample output preview ─────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Sample HIGH risk scored events:")
print("=" * 55)
high_risk = df[df["risk_label"] == "HIGH"][
    ["user_name", "anomaly_type", "final_risk_score",
     "iso_score", "xgb_high_prob", "shap_explanation"]
].head(5)
for _, row in high_risk.iterrows():
    print(f"\n  User       : {row['user_name']}")
    print(f"  Anomaly    : {row['anomaly_type']}")
    print(f"  Risk Score : {row['final_risk_score']}/100")
    print(f"  ISO Score  : {row['iso_score']}")
    print(f"  XGB HIGH % : {row['xgb_high_prob']:.2%}")
    explanation = json.loads(row["shap_explanation"])
    print(f"  Top reasons:")
    for e in explanation:
        direction = "↑ increases risk" if e["shap"] > 0 else "↓ decreases risk"
        print(f"    - {e['feature']:25s} {direction}  (SHAP: {e['shap']:+.4f})")