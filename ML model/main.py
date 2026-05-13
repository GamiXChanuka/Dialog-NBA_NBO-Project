"""
Dialog NBA/NBO - Random Forest Classifier
==========================================
Trains two models:
  1. Action Model  → predicts predicted_best_action
  2. Offer Model   → predicts predicted_offer

Usage:
  python dialog_nba_nbo_model.py          # train + save models
  python dialog_nba_nbo_model.py predict  # interactive prediction prompt
"""

import sys
import os
import io
import warnings
import numpy as np
import pandas as pd

# Ensure stdout uses UTF-8 to prevent UnicodeEncodeError on Windows
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
import joblib
import json

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_PATH        = os.path.join(BASE_DIR, "dialog_nba_nbo_expanded.csv")
ACTION_MODEL     = os.path.join(BASE_DIR, "model_action.pkl")
OFFER_MODEL      = os.path.join(BASE_DIR, "model_offer.pkl")
ENCODERS_FILE    = os.path.join(BASE_DIR, "label_encoders.pkl")
FEATURE_META     = os.path.join(BASE_DIR, "feature_meta.json")

# ─── Feature columns used for training ───────────────────────────────────────
CATEGORICAL_COLS = [
    "district", "age_group", "current_package",
    "device_type", "sim_type", "previous_offer_response"
]

NUMERIC_COLS = [
    "monthly_data_usage_gb", "youtube_usage_gb", "social_usage_gb",
    "voice_minutes", "sms_usage", "monthly_spend_lkr",
    "reload_frequency", "add_on_count", "is_5g_supported",
    "router_owned", "churn_risk_score", "complaint_count",
    "days_since_last_change"
]

FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS
TARGET_ACTION = "predicted_best_action"
TARGET_OFFER  = "predicted_offer"


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[1/5] Cleaning data...")
    original_shape = df.shape

    # Drop non-feature columns
    drop_cols = ["customer_id", "accepted"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Fill missing numerics with median
    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
            print(f"   • Filled {col} nulls with median")

    # Fill missing categoricals with mode
    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
            print(f"   • Filled {col} nulls with mode")

    # Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    dupes = before - len(df)
    if dupes:
        print(f"   • Removed {dupes} duplicate rows")

    # Clip outliers in numeric cols to [1st, 99th] percentile
    for col in NUMERIC_COLS:
        if col in df.columns:
            lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = df[col].clip(lo, hi)

    print(f"   ✓ Shape: {original_shape} → {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[2/5] Engineering features...")

    # Usage ratios
    df["social_ratio"]   = df["social_usage_gb"]  / (df["monthly_data_usage_gb"] + 1e-6)
    df["youtube_ratio"]  = df["youtube_usage_gb"]  / (df["monthly_data_usage_gb"] + 1e-6)

    # Package usage saturation (how full is the plan?)
    pkg_map = {"10GB": 10, "20GB": 20, "50GB": 50, "100GB": 100}
    df["pkg_limit"]      = df["current_package"].map(pkg_map).fillna(50)
    df["usage_saturation"] = df["monthly_data_usage_gb"] / df["pkg_limit"]

    # Spend per GB
    df["spend_per_gb"]   = df["monthly_spend_lkr"] / (df["monthly_data_usage_gb"] + 1e-6)

    # High social flag
    df["high_social"]    = (df["social_ratio"] > 0.20).astype(int)

    # Heavy voice flag
    df["heavy_voice"]    = (df["voice_minutes"] > 1500).astype(int)

    # Data-hungry flag
    df["data_hungry"]    = (df["usage_saturation"] > 0.85).astype(int)

    # Is at-risk customer
    df["at_risk"]        = (df["churn_risk_score"] > 0.60).astype(int)

    print("   ✓ Added: social_ratio, youtube_ratio, usage_saturation,",
          "spend_per_gb, high_social, heavy_voice, data_hungry, at_risk")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. ENCODING
# ══════════════════════════════════════════════════════════════════════════════
def encode_features(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    print("\n[3/5] Encoding categorical features...")

    if fit:
        encoders = {}

    # All categorical cols (original + pkg_limit is already numeric)
    cat_cols = CATEGORICAL_COLS + []   # engineered cols are already numeric
    for col in cat_cols:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # Handle unseen labels gracefully
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    print(f"   ✓ Encoded {len(cat_cols)} categorical columns")
    return df, encoders


def get_all_feature_cols(df: pd.DataFrame):
    base = FEATURE_COLS.copy()
    engineered = ["social_ratio", "youtube_ratio", "usage_saturation",
                  "spend_per_gb", "high_social", "heavy_voice",
                  "data_hungry", "at_risk", "pkg_limit"]
    return [c for c in base + engineered if c in df.columns]


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════════
def train_model(X_train, y_train, label: str) -> RandomForestClassifier:
    print(f"\n   Training {label} model...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",   # handles class imbalance
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, label: str, le: LabelEncoder):
    print(f"\n─── {label} Model Evaluation ───────────────────────────")
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   F1-Score : {f1:.4f}")
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_test, y_test, cv=cv,
                                scoring="f1_weighted", n_jobs=-1)
    print(f"   5-Fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Top 10 feature importances
    feat_names = model.feature_names_in_
    importances = pd.Series(model.feature_importances_, index=feat_names)
    print("\n   Top 10 Feature Importances:")
    print(importances.sort_values(ascending=False).head(10).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def train_pipeline():
    print("=" * 60)
    print("  Dialog NBA/NBO - Random Forest Training Pipeline")
    print("=" * 60)

    # Load
    print(f"\n[0/5] Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"   ✓ Loaded {len(df):,} rows × {df.shape[1]} columns")

    # Clean
    df = clean_data(df)

    # Engineer
    df = add_features(df)

    # Encode features (fit encoders on training data)
    df, encoders = encode_features(df, fit=True)

    # Encode targets
    le_action = LabelEncoder()
    le_offer  = LabelEncoder()
    y_action  = le_action.fit_transform(df[TARGET_ACTION])
    y_offer   = le_offer.fit_transform(df[TARGET_OFFER])
    encoders["__target_action__"] = le_action
    encoders["__target_offer__"]  = le_offer

    # Feature matrix
    feat_cols = get_all_feature_cols(df)
    X = df[feat_cols]

    print(f"\n[4/5] Splitting data (80/20 stratified split)...")
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
        X, y_action, test_size=0.2, stratify=y_action, random_state=42)
    X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
        X, y_offer, test_size=0.2, stratify=y_offer, random_state=42)

    print(f"   ✓ Train: {len(X_train_a):,}  Test: {len(X_test_a):,}")

    # Train
    model_action = train_model(X_train_a, y_train_a, "Action")
    model_offer  = train_model(X_train_o, y_train_o, "Offer")

    # Evaluate
    print("\n[5/5] Evaluating models...")
    evaluate_model(model_action, X_test_a, y_test_a, "Action", le_action)
    evaluate_model(model_offer,  X_test_o, y_test_o, "Offer",  le_offer)

    # Save
    joblib.dump(model_action, ACTION_MODEL)
    joblib.dump(model_offer,  OFFER_MODEL)
    joblib.dump(encoders,     ENCODERS_FILE)

    meta = {"feature_cols": feat_cols}
    with open(FEATURE_META, "w") as f:
        json.dump(meta, f)

    print("\n" + "=" * 60)
    print("  Models saved:")
    print(f"    → {ACTION_MODEL}")
    print(f"    → {OFFER_MODEL}")
    print(f"    → {ENCODERS_FILE}")
    print(f"    → {FEATURE_META}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# 6. PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
def predict(user_input: dict) -> dict:
    """
    Predict best action and best offer for a single customer.

    Parameters
    ----------
    user_input : dict with keys matching FEATURE_COLS

    Returns
    -------
    dict with keys: predicted_best_action, action_confidence,
                    predicted_offer, offer_confidence,
                    top3_actions, top3_offers
    """
    # Load artefacts
    model_action = joblib.load(ACTION_MODEL)
    model_offer  = joblib.load(OFFER_MODEL)
    encoders     = joblib.load(ENCODERS_FILE)
    with open(FEATURE_META) as f:
        meta = json.load(f)

    le_action = encoders["__target_action__"]
    le_offer  = encoders["__target_offer__"]
    feat_cols = meta["feature_cols"]

    # Build DataFrame from input
    row = pd.DataFrame([user_input])

    # Engineer same features
    row = add_features(row)

    # Encode categoricals
    row, _ = encode_features(row, encoders=encoders, fit=False)

    # Align columns (fill any missing engineered cols with 0)
    for col in feat_cols:
        if col not in row.columns:
            row[col] = 0
    row = row[feat_cols]

    # Predict action
    action_proba  = model_action.predict_proba(row)[0]
    action_idx    = np.argmax(action_proba)
    action_label  = le_action.inverse_transform([action_idx])[0]
    action_conf   = round(float(action_proba[action_idx]) * 100, 2)
    top3_actions  = [
        {"action": le_action.inverse_transform([i])[0],
         "confidence": round(float(p)*100, 2)}
        for i, p in sorted(enumerate(action_proba),
                           key=lambda x: -x[1])[:3]
    ]

    # Predict offer
    offer_proba   = model_offer.predict_proba(row)[0]
    offer_idx     = np.argmax(offer_proba)
    offer_label   = le_offer.inverse_transform([offer_idx])[0]
    offer_conf    = round(float(offer_proba[offer_idx]) * 100, 2)
    top3_offers   = [
        {"offer": le_offer.inverse_transform([i])[0],
         "confidence": round(float(p)*100, 2)}
        for i, p in sorted(enumerate(offer_proba),
                           key=lambda x: -x[1])[:3]
    ]

    return {
        "predicted_best_action" : action_label,
        "action_confidence_%"   : action_conf,
        "top3_actions"          : top3_actions,
        "predicted_offer"       : offer_label,
        "offer_confidence_%"    : offer_conf,
        "top3_offers"           : top3_offers,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. INTERACTIVE CLI PROMPT
# ══════════════════════════════════════════════════════════════════════════════
def interactive_prompt():
    print("\n" + "=" * 60)
    print("  Dialog NBA/NBO - Prediction Mode")
    print("  (Enter customer details to get recommendations)")
    print("=" * 60)

    def ask(prompt, default, cast=str):
        val = input(f"  {prompt} [{default}]: ").strip()
        return cast(val) if val else cast(default)

    while True:
        print("\n── Customer Input ──────────────────────────────────")

        inp = {
            "district"               : ask("District (e.g. Colombo, Kandy, Galle)", "Colombo"),
            "age_group"              : ask("Age group (18-25/26-35/36-45/46-60/60+)", "26-35"),
            "current_package"        : ask("Current package (10GB/20GB/50GB/100GB)", "20GB"),
            "monthly_data_usage_gb"  : ask("Monthly data usage (GB)", 35.0, float),
            "youtube_usage_gb"       : ask("YouTube usage (GB)", 10.0, float),
            "social_usage_gb"        : ask("Social media usage (GB)", 8.0, float),
            "voice_minutes"          : ask("Voice minutes / month", 500, int),
            "sms_usage"              : ask("SMS count / month", 100, int),
            "monthly_spend_lkr"      : ask("Monthly spend (LKR)", 5000, int),
            "reload_frequency"       : ask("Reload frequency (times/month)", 10, int),
            "add_on_count"           : ask("Active add-ons count", 2, int),
            "device_type"            : ask("Device type (Basic_Phone/4G_Mobile/5G_Mobile)", "4G_Mobile"),
            "is_5g_supported"        : ask("Device 5G supported? (1=Yes, 0=No)", 0, int),
            "sim_type"               : ask("SIM type (4G/5G)", "4G"),
            "router_owned"           : ask("Owns router? (1=Yes, 0=No)", 0, int),
            "churn_risk_score"       : ask("Churn risk score (0.0–1.0)", 0.3, float),
            "complaint_count"        : ask("Complaint count", 1, int),
            "previous_offer_response": ask("Previous offer response (Accepted/Ignored/Rejected)", "Ignored"),
            "days_since_last_change" : ask("Days since last plan change", 180, int),
        }

        result = predict(inp)

        print("\n── Prediction Results ──────────────────────────────")
        print(f"  🎯 Best Action : {result['predicted_best_action']}")
        print(f"     Confidence  : {result['action_confidence_%']}%")
        print(f"  🎁 Best Offer  : {result['predicted_offer']}")
        print(f"     Confidence  : {result['offer_confidence_%']}%")
        print("\n  Top 3 Actions:")
        for i, a in enumerate(result["top3_actions"], 1):
            print(f"    {i}. {a['action']} ({a['confidence']}%)")
        print("\n  Top 3 Offers:")
        for i, o in enumerate(result["top3_offers"], 1):
            print(f"    {i}. {o['offer']} ({o['confidence']}%)")

        again = input("\n  Predict another customer? (y/n) [y]: ").strip().lower()
        if again == "n":
            print("\n  Goodbye!\n")
            break


# ══════════════════════════════════════════════════════════════════════════════
# 8. QUICK PREDICT DEMO (no prompts)
# ══════════════════════════════════════════════════════════════════════════════
def demo_predict():
    """Run a few hardcoded examples to verify the models work."""
    examples = [
        {
            "label": "High social user on 10GB plan",
            "input": {
                "district": "Colombo", "age_group": "26-35",
                "current_package": "10GB", "monthly_data_usage_gb": 18.0,
                "youtube_usage_gb": 3.0, "social_usage_gb": 14.0,
                "voice_minutes": 300, "sms_usage": 50,
                "monthly_spend_lkr": 3000, "reload_frequency": 12,
                "add_on_count": 1, "device_type": "4G_Mobile",
                "is_5g_supported": 0, "sim_type": "4G",
                "router_owned": 0, "churn_risk_score": 0.4,
                "complaint_count": 0, "previous_offer_response": "Accepted",
                "days_since_last_change": 90,
            }
        },
        {
            "label": "Heavy data user, no router",
            "input": {
                "district": "Gampaha", "age_group": "36-45",
                "current_package": "50GB", "monthly_data_usage_gb": 85.0,
                "youtube_usage_gb": 30.0, "social_usage_gb": 10.0,
                "voice_minutes": 700, "sms_usage": 200,
                "monthly_spend_lkr": 12000, "reload_frequency": 5,
                "add_on_count": 3, "device_type": "5G_Mobile",
                "is_5g_supported": 1, "sim_type": "5G",
                "router_owned": 0, "churn_risk_score": 0.25,
                "complaint_count": 1, "previous_offer_response": "Accepted",
                "days_since_last_change": 200,
            }
        },
        {
            "label": "High churn risk customer",
            "input": {
                "district": "Jaffna", "age_group": "46-60",
                "current_package": "20GB", "monthly_data_usage_gb": 15.0,
                "youtube_usage_gb": 5.0, "social_usage_gb": 3.0,
                "voice_minutes": 400, "sms_usage": 80,
                "monthly_spend_lkr": 2500, "reload_frequency": 3,
                "add_on_count": 0, "device_type": "Basic_Phone",
                "is_5g_supported": 0, "sim_type": "4G",
                "router_owned": 0, "churn_risk_score": 0.82,
                "complaint_count": 5, "previous_offer_response": "Rejected",
                "days_since_last_change": 450,
            }
        },
    ]

    print("\n── Demo Predictions ────────────────────────────────────")
    for ex in examples:
        print(f"\n📋 Scenario: {ex['label']}")
        r = predict(ex["input"])
        print(f"   🎯 Action : {r['predicted_best_action']} ({r['action_confidence_%']}%)")
        print(f"   🎁 Offer  : {r['predicted_offer']} ({r['offer_confidence_%']}%)")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    if mode == "train":
        train_pipeline()
        print("\nRunning demo predictions on trained models...")
        demo_predict()

    elif mode == "predict":
        if not os.path.exists(ACTION_MODEL):
            print("  ⚠  Models not found. Run training first:")
            print("     python dialog_nba_nbo_model.py")
            sys.exit(1)
        interactive_prompt()

    elif mode == "demo":
        demo_predict()

    else:
        print("Usage: python dialog_nba_nbo_model.py [train|predict|demo]")