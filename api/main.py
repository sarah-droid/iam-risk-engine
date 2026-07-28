from api.email_alerts import load_admin_emails, add_admin_email, remove_admin_email, send_alert_email
from fastapi import File, UploadFile
import io
import json
import sqlite3
import joblib
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="IdentityGuard AI — IAM Risk Engine",
    description="AI-powered Identity Threat Detection & Access Risk Scoring API",
    version="1.0.0",
)

# Allow dashboard (HTML file) to call this API from browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS + METADATA ON STARTUP
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "ml_models")
DATA_DIR   = os.path.join(BASE_DIR, "datasets")
DB_PATH    = os.path.join(DATA_DIR, "audit.db")

print("Loading ML models ...")
iso_forest  = joblib.load(os.path.join(MODELS_DIR, "isolation_forest.pkl"))
xgb_model   = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
shap_explainer = joblib.load(os.path.join(MODELS_DIR, "shap_explainer.pkl"))

with open(os.path.join(MODELS_DIR, "model_meta.json")) as f:
    meta = json.load(f)

FEATURE_COLS  = meta["feature_cols"]
LABEL_MAP_INV = {int(k): v for k, v in meta["label_map_inv"].items()}
ISO_MIN       = meta["iso_min"]
ISO_MAX       = meta["iso_max"]
ISO_WEIGHT    = meta["iso_weight"]
XGB_WEIGHT    = meta["xgb_weight"]

print("✅ Models loaded successfully")

# Policy thresholds
POLICY_RULES = [
    {"min": 80, "max": 100, "action": "BLOCK",        "action_ja": "アクセス拒否"},
    {"min": 60, "max": 79,  "action": "FORCE_MFA",    "action_ja": "多要素認証を強制"},
    {"min": 40, "max": 59,  "action": "NOTIFY_ADMIN", "action_ja": "管理者に通知"},
    {"min": 0,  "max": 39,  "action": "ALLOW",        "action_ja": "アクセスを許可"},
]

def score_to_action(score):
    for rule in POLICY_RULES:
        if rule["min"] <= score <= rule["max"]:
            return rule["action"], rule["action_ja"]
    return "ALLOW", "アクセスを許可"

def write_audit(data: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO audit_log
        (timestamp, session_id, user_id, user_name, department, country,
         anomaly_type, risk_score, risk_label, action_taken, action_ja,
         severity, top_reasons, triggered_by)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.utcnow().isoformat(),
        data.get("session_id",""),
        data.get("user_id",""),
        data.get("user_name",""),
        data.get("department",""),
        data.get("country",""),
        data.get("anomaly_type","none"),
        data.get("risk_score", 0),
        data.get("risk_label","LOW"),
        data.get("action","ALLOW"),
        data.get("action_ja","アクセスを許可"),
        data.get("severity","LOW"),
        data.get("top_reasons",""),
        data.get("triggered_by","API"),
    ))
    conn.commit()
    conn.close()

# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ScoreRequest(BaseModel):
    user_id:           str
    user_name:         str
    department:        Optional[str] = "Unknown"
    country:           Optional[str] = "Japan"
    city:              Optional[str] = "Tokyo"
    impossible_travel: Optional[int] = 0
    hours_since_last:  Optional[float] = 8.0
    mfa_fail_count:    Optional[int] = 0
    is_known_device:   Optional[int] = 1
    country_deviation: Optional[int] = 0
    hour_sin:          Optional[float] = 0.0
    hour_cos:          Optional[float] = 1.0
    is_business_hours: Optional[int] = 1
    role_rank:         Optional[int] = 0
    role_escalation:   Optional[int] = 0
    pwd_reset_7d:      Optional[int] = 0
    peer_zscore:       Optional[float] = 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "app":     "IdentityGuard AI",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "/docs",
    }

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# ── GET /events ───────────────────────────────────────────────────────────────
@app.get("/events")
def get_events(limit: int = 50, risk: Optional[str] = None):
    """
    Returns recent scored IAM events.
    Optional filter: ?risk=HIGH or ?risk=MEDIUM or ?risk=LOW
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "scored_events.csv"))
    df = df.sort_values("timestamp", ascending=False)

    if risk:
        df = df[df["risk_label"].str.upper() == risk.upper()]

    df = df.head(limit)

    events = []
    for _, row in df.iterrows():
        try:
            shap_data = json.loads(str(row.get("shap_explanation", "[]")))
        except Exception:
            shap_data = []

        events.append({
            "session_id":     str(row.get("session_id", "")),
            "user_id":        row["user_id"],
            "user_name":      row["user_name"],
            "department":     row["department"],
            "timestamp":      row["timestamp"],
            "country":        row["country"],
            "city":           row["city"],
            "anomaly_type":   row["anomaly_type"],
            "risk_label":     row["risk_label"],
            "risk_score":     int(row["final_risk_score"]),
            "action":         row.get("action_taken", "ALLOW"),
            "action_ja":      row.get("action_ja", "アクセスを許可"),
            "iso_score":      round(float(row.get("iso_score", 0)), 1),
            "xgb_high_prob":  round(float(row.get("xgb_high_prob", 0)), 4),
            "role_rank":      int(row.get("role_rank", 0)) if str(row.get("role_rank","")).strip() != "" else 0,
            "explanation":    shap_data,
        })

    return {"total": len(events), "events": events}


# ── POST /score ───────────────────────────────────────────────────────────────
@app.post("/score")
def score_event(req: ScoreRequest):
    """
    Accepts a login event, runs it through ML models,
    returns risk score + action + explanation.
    """
    features = [[
        req.impossible_travel,
        req.hours_since_last,
        req.mfa_fail_count,
        req.is_known_device,
        req.country_deviation,
        req.hour_sin,
        req.hour_cos,
        req.is_business_hours,
        req.role_rank,
        req.role_escalation,
        req.pwd_reset_7d,
        req.peer_zscore,
    ]]

    X = np.array(features)

    # Isolation Forest score
    iso_raw = float(iso_forest.decision_function(X)[0])
    iso_normalised = 100 * (1 - (iso_raw - ISO_MIN) / (ISO_MAX - ISO_MIN))
    iso_normalised = float(np.clip(iso_normalised, 0, 100))

    # XGBoost probabilities
    xgb_proba = xgb_model.predict_proba(X)[0]
    xgb_pred  = int(xgb_model.predict(X)[0])
    high_prob = float(xgb_proba[2])

    # Combined score
    final_score = int(np.clip(
        ISO_WEIGHT * iso_normalised + XGB_WEIGHT * (high_prob * 100),
        0, 100
    ))

    risk_label = LABEL_MAP_INV.get(xgb_pred, "LOW")
    action, action_ja = score_to_action(final_score)

    # SHAP explanation
    X_df = pd.DataFrame(X, columns=FEATURE_COLS)
    shap_vals = shap_explainer.shap_values(X_df)
    if isinstance(shap_vals, list):
        row_shap = shap_vals[xgb_pred][0]
    else:
        row_shap = shap_vals[0, :, xgb_pred]

    feat_importance = sorted(
        zip(FEATURE_COLS, row_shap),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    explanation = [
        {
            "feature":   f,
            "shap":      round(float(s), 4),
            "direction": "increases risk" if s > 0 else "decreases risk"
        }
        for f, s in feat_importance[:3]
    ]

    # Write to audit log
    write_audit({
        "user_id":    req.user_id,
        "user_name":  req.user_name,
        "department": req.department,
        "country":    req.country,
        "risk_score": final_score,
        "risk_label": risk_label,
        "action":     action,
        "action_ja":  action_ja,
        "severity":   risk_label,
        "top_reasons": json.dumps(explanation),
        "triggered_by": "API_SCORE_ENDPOINT",
    })

    return {
        "user_id":      req.user_id,
        "user_name":    req.user_name,
        "risk_score":   final_score,
        "risk_label":   risk_label,
        "action":       action,
        "action_ja":    action_ja,
        "iso_score":    round(iso_normalised, 1),
        "xgb_high_prob": round(high_prob, 4),
        "explanation":  explanation,
        "timestamp":    datetime.utcnow().isoformat(),
    }


# ── GET /audit ────────────────────────────────────────────────────────────────
@app.get("/audit")
def get_audit(limit: int = 50):
    """Returns recent audit log entries."""
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(
        f"SELECT * FROM audit_log ORDER BY id DESC LIMIT {limit}", conn
    )
    conn.close()
    return {"total": len(df), "logs": df.to_dict(orient="records")}


# ── GET /stats ────────────────────────────────────────────────────────────────
@app.get("/stats")
def get_stats():
    """Returns summary statistics for the dashboard metrics panel."""
    df = pd.read_csv(os.path.join(DATA_DIR, "scored_events.csv"))
    return {
        "total_events": len(df),
        "high_risk":    int((df["risk_label"] == "HIGH").sum()),
        "medium_risk":  int((df["risk_label"] == "MEDIUM").sum()),
        "low_risk":     int((df["risk_label"] == "LOW").sum()),
        "blocked":      int((df.get("action_taken","") == "BLOCK").sum()),
        "mfa_forced":   int((df.get("action_taken","") == "FORCE_MFA").sum()),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# ADMIN EMAIL ALERTS
# Paste this entire block at the END of your existing api/main.py
# (after your /stats endpoint, before the final blank line)
#
# Also add this import near the top of main.py, with your other imports:
#     from api.email_alerts import (
#         load_admin_emails, add_admin_email, remove_admin_email, send_alert_email
#     )
#
# If main.py is INSIDE the api/ folder and you run uvicorn from the project
# root with `python -m uvicorn api.main:app`, use this import instead:
#     from email_alerts import (
#         load_admin_emails, add_admin_email, remove_admin_email, send_alert_email
#     )
# ═══════════════════════════════════════════════════════════════════════════════

from pydantic import BaseModel as _BaseModel  # safe even if BaseModel already imported above


class AdminEmailRequest(_BaseModel):
    name: str
    email: str


class AlertAdminRequest(_BaseModel):
    user_id: str = "manual_trigger"
    user_name: str = "Manual Alert"
    risk_score: int = 50
    risk_label: str = "MEDIUM"
    action: str = "NOTIFY_ADMIN"
    anomaly_type: str = "manual_alert"
    country: str = "Unknown"
    reasons: list = []


# ── GET /admins ──────────────────────────────────────────────────────────────
@app.get("/admins")
def get_admins():
    """Returns the list of admin email addresses configured to receive alerts."""
    return {"admins": load_admin_emails()}


# ── POST /admins ─────────────────────────────────────────────────────────────
@app.post("/admins")
def add_admin(req: AdminEmailRequest):
    """Adds a new admin email address to the alert recipient list."""
    if "@" not in req.email or "." not in req.email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Invalid email address")

    admins, added = add_admin_email(req.name, req.email)
    if not added:
        raise HTTPException(status_code=409, detail="This email is already registered")
    return {"admins": admins, "message": f"Added {req.email}"}


# ── DELETE /admins/{email} ────────────────────────────────────────────────────
@app.delete("/admins/{email}")
def delete_admin(email: str):
    """Removes an admin email address from the alert recipient list."""
    admins, removed = remove_admin_email(email)
    if not removed:
        raise HTTPException(status_code=404, detail="Email not found")
    return {"admins": admins, "message": f"Removed {email}"}


# ── POST /alert-admin ──────────────────────────────────────────────────────────
@app.post("/alert-admin")
def alert_admin(req: AlertAdminRequest):
    """
    Sends a real email alert to every registered admin address.
    Used by the dashboard's "🔔 Alert Admin" button, and can also be called
    automatically by the policy engine when a HIGH risk event is detected.
    """
    event_data = {
        "user_id":      req.user_id,
        "user_name":    req.user_name,
        "risk_score":   req.risk_score,
        "risk_label":   req.risk_label,
        "action":       req.action,
        "anomaly_type": req.anomaly_type,
        "country":      req.country,
        "timestamp":    datetime.utcnow().isoformat(),
        "reasons":      req.reasons,
    }

    success, message, sent_to = send_alert_email(event_data)

    if not success:
        raise HTTPException(status_code=500, detail=message)

    return {
        "success": True,
        "message": message,
        "sent_to": sent_to,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# ADMIN EMAIL ALERTS
# Paste this entire block at the END of your existing api/main.py
# (after your /stats endpoint, before the final blank line)
#
# Also add this import near the top of main.py, with your other imports:
#     from api.email_alerts import (
#         load_admin_emails, add_admin_email, remove_admin_email, send_alert_email
#     )
#
# If main.py is INSIDE the api/ folder and you run uvicorn from the project
# root with `python -m uvicorn api.main:app`, use this import instead:
#     from email_alerts import (
#         load_admin_emails, add_admin_email, remove_admin_email, send_alert_email
#     )
# ═══════════════════════════════════════════════════════════════════════════════

from pydantic import BaseModel as _BaseModel  # safe even if BaseModel already imported above


class AdminEmailRequest(_BaseModel):
    name: str
    email: str


class AlertAdminRequest(_BaseModel):
    user_id: str = "manual_trigger"
    user_name: str = "Manual Alert"
    risk_score: int = 50
    risk_label: str = "MEDIUM"
    action: str = "NOTIFY_ADMIN"
    anomaly_type: str = "manual_alert"
    country: str = "Unknown"
    reasons: list = []


# ── GET /admins ──────────────────────────────────────────────────────────────
@app.get("/admins")
def get_admins():
    """Returns the list of admin email addresses configured to receive alerts."""
    return {"admins": load_admin_emails()}


# ── POST /admins ─────────────────────────────────────────────────────────────
@app.post("/admins")
def add_admin(req: AdminEmailRequest):
    """Adds a new admin email address to the alert recipient list."""
    if "@" not in req.email or "." not in req.email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Invalid email address")

    admins, added = add_admin_email(req.name, req.email)
    if not added:
        raise HTTPException(status_code=409, detail="This email is already registered")
    return {"admins": admins, "message": f"Added {req.email}"}


# ── DELETE /admins/{email} ────────────────────────────────────────────────────
@app.delete("/admins/{email}")
def delete_admin(email: str):
    """Removes an admin email address from the alert recipient list."""
    admins, removed = remove_admin_email(email)
    if not removed:
        raise HTTPException(status_code=404, detail="Email not found")
    return {"admins": admins, "message": f"Removed {email}"}


# ── POST /alert-admin ──────────────────────────────────────────────────────────
@app.post("/alert-admin")
def alert_admin(req: AlertAdminRequest):
    """
    Sends a real email alert to every registered admin address.
    Used by the dashboard's "🔔 Alert Admin" button, and can also be called
    automatically by the policy engine when a HIGH risk event is detected.
    """
    event_data = {
        "user_id":      req.user_id,
        "user_name":    req.user_name,
        "risk_score":   req.risk_score,
        "risk_label":   req.risk_label,
        "action":       req.action,
        "anomaly_type": req.anomaly_type,
        "country":      req.country,
        "timestamp":    datetime.utcnow().isoformat(),
        "reasons":      req.reasons,
    }

    success, message, sent_to = send_alert_email(event_data)

    if not success:
        raise HTTPException(status_code=500, detail=message)

    return {
        "success": True,
        "message": message,
        "sent_to": sent_to,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# KEYCLOAK USER PROVISIONING
# Also add this import near the top of main.py:
#     import requests
# (already a dependency since collector/keycloak_collector.py uses it)
# ═══════════════════════════════════════════════════════════════════════════════

import requests as _requests  # aliased to avoid clashing if `requests` is already imported differently above

KEYCLOAK_BASE_URL = "http://localhost:8080"
KEYCLOAK_ADMIN_USER = "admin"
KEYCLOAK_ADMIN_PASSWORD = "admin"  # matches the docker run command used to start Keycloak


class ProvisionKeycloakUserRequest(_BaseModel):
    name: str
    email: str
    department: str = "Unknown"
    role: str = "viewer"
    realm: str = "master"


def _get_keycloak_admin_token():
    """Authenticates against Keycloak's admin-cli and returns an access token."""
    resp = _requests.post(
        f"{KEYCLOAK_BASE_URL}/realms/master/protocol/openid-connect/token",
        data={
            "client_id": "admin-cli",
            "username": KEYCLOAK_ADMIN_USER,
            "password": KEYCLOAK_ADMIN_PASSWORD,
            "grant_type": "password",
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


# ── POST /provision-keycloak-user ───────────────────────────────────────────
@app.post("/provision-keycloak-user")
def provision_keycloak_user(req: ProvisionKeycloakUserRequest):
    """
    Creates a real user in your local Keycloak instance via the Admin REST API.
    This is the one identity source in the "Invite User" modal that's actually
    wired to a real, callable IAM system — Okta and Azure AD fields in the
    dashboard are intentionally UI-only, since those need a live tenant and
    credentials this local demo environment doesn't have.
    """
    name_parts = req.name.strip().split(" ", 1)
    first_name = name_parts[0]
    last_name = name_parts[1] if len(name_parts) > 1 else ""

    # Derive a username from the email's local part, lowercased
    username = req.email.split("@")[0].lower()

    try:
        token = _get_keycloak_admin_token()
    except _requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Could not authenticate with Keycloak: {e}")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    user_payload = {
        "username": username,
        "email": req.email,
        "firstName": first_name,
        "lastName": last_name,
        "enabled": True,
        "emailVerified": False,
        "attributes": {
            "department": [req.department],
            "initial_role": [req.role],
        },
        "credentials": [{
            "type": "password",
            "value": "ChangeMe123!",   # temporary onboarding password
            "temporary": True,         # forces password reset on first login
        }],
    }

    try:
        create_resp = _requests.post(
            f"{KEYCLOAK_BASE_URL}/admin/realms/{req.realm}/users",
            headers=headers,
            json=user_payload,
            timeout=10,
        )
    except _requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Could not reach Keycloak: {e}")

    if create_resp.status_code == 201:
        # Keycloak returns the new user's location in a Location header,
        # not in the response body — extract the user ID from it.
        location = create_resp.headers.get("Location", "")
        user_id = location.rstrip("/").split("/")[-1] if location else None
        return {
            "success": True,
            "message": f"User '{username}' created in Keycloak realm '{req.realm}'",
            "username": username,
            "user_id": user_id,
        }
    elif create_resp.status_code == 409:
        raise HTTPException(status_code=409, detail=f"A user with username '{username}' or email '{req.email}' already exists in Keycloak")
    else:
        raise HTTPException(
            status_code=502,
            detail=f"Keycloak returned an error ({create_resp.status_code}): {create_resp.text[:200]}"
        )

# ═══════════════════════════════════════════════════════════════════════════════
# UPLOAD ANALYSIS ENDPOINT
# Paste this section at the END of your existing api/main.py
# (after all your existing endpoints)
#
# Also add these imports near the top of main.py if not already present:
#     from fastapi import File, UploadFile
#     import io
# ═══════════════════════════════════════════════════════════════════════════════

from fastapi import File, UploadFile
import io as _io

# ── POST /analyse-upload ──────────────────────────────────────────────────────
@app.post("/analyse-upload")
async def analyse_upload(file: UploadFile = File(...)):
    """
    Accepts a CSV or Excel file containing raw IAM login events,
    runs feature engineering and ML risk scoring on it,
    and returns a full scored dataset plus summary statistics.

    Accepted column names (flexible — extra columns are ignored,
    missing optional columns get sensible defaults):

    Required:
        user_id, timestamp

    Optional (used for ML features if present):
        user_name, department, country, city, latitude, longitude,
        hour_of_day, role_rank, mfa_failed, mfa_success, password_reset,
        is_known_device, device_id, anomaly_type, risk_label

    The endpoint re-uses the trained models saved in ml_models/ —
    no retraining happens, only inference on the new data.
    """
    # ── Read uploaded file ────────────────────────────────────────────────
    contents = await file.read()
    filename  = file.filename or ""

    try:
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            df = pd.read_excel(_io.BytesIO(contents))
        elif filename.endswith(".csv") or filename.endswith(".txt"):
            df = pd.read_csv(_io.StringIO(contents.decode("utf-8", errors="replace")))
        else:
            # Try CSV as fallback regardless of extension
            try:
                df = pd.read_csv(_io.StringIO(contents.decode("utf-8", errors="replace")))
            except Exception:
                raise HTTPException(status_code=400, detail="Could not parse file. Please upload a CSV or Excel (.xlsx) file.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(df) < 1:
        raise HTTPException(status_code=400, detail="File must contain at least one data row.")

    # Normalise column names to lowercase with underscores
    df.columns = [c.strip().lower().replace(" ","_").replace("-","_") for c in df.columns]

    # ── Require at minimum: user_id ───────────────────────────────────────
    if "user_id" not in df.columns and "userid" not in df.columns and "user" not in df.columns and "email" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="File must contain at least a 'user_id' (or 'user' / 'email') column. "
                   "Accepted columns: user_id, user_name, department, country, timestamp, "
                   "hour_of_day, role_rank, mfa_failed, is_known_device, password_reset, anomaly_type"
        )

    # ── Column alias normalisation ────────────────────────────────────────
    aliases = {
        "userid": "user_id", "user": "user_id", "email": "user_id",
        "username": "user_name", "name": "user_name",
        "dept": "department", "team": "department",
        "ts": "timestamp", "date": "timestamp", "time": "timestamp",
        "hour": "hour_of_day",
        "role": "role_rank",
        "mfa_fail": "mfa_failed", "mfa_failures": "mfa_failed",
        "known_device": "is_known_device",
        "pwd_reset": "password_reset",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})

    # ── Fill missing optional columns with sensible defaults ──────────────
    defaults = {
        "user_name":       lambda: df.get("user_id", pd.Series(["Unknown"]*len(df))),
        "department":      "Unknown",
        "country":         "Unknown",
        "city":            "Unknown",
        "latitude":        0.0,
        "longitude":       0.0,
        "hour_of_day":     12,
        "role_rank":       0,
        "mfa_failed":      0,
        "mfa_success":     1,
        "password_reset":  0,
        "is_known_device": 1,
        "anomaly_type":    "none",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default(df) if callable(default) else default

    # ── Parse timestamps if present ───────────────────────────────────────
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.Timestamp.now().isoformat()
    else:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["timestamp"] = df["timestamp"].fillna(pd.Timestamp.now())
            if "hour_of_day" not in df.columns or df["hour_of_day"].eq(12).all():
                df["hour_of_day"] = df["timestamp"].apply(lambda t: t.hour if hasattr(t,"hour") else 12)
            df["timestamp"] = df["timestamp"].astype(str)
        except Exception:
            df["timestamp"] = pd.Timestamp.now().isoformat()

    # ── Feature engineering (simplified inline version) ───────────────────
    import numpy as _np
    from math import radians, sin, cos, sqrt, atan2

    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1-a))

    user_country_history = df.groupby("user_id")["country"].apply(set).to_dict()
    user_role_median     = df.groupby("user_id")["role_rank"].median().to_dict()

    feature_rows = []
    last_event   = {}

    for _, row in df.iterrows():
        uid  = str(row.get("user_id", "unknown"))
        prev = last_event.get(uid)

        # Impossible travel
        impossible = 0
        if prev is not None:
            try:
                dist = haversine_km(float(prev.get("latitude",0)), float(prev.get("longitude",0)),
                                    float(row.get("latitude",0)),  float(row.get("longitude",0)))
                hours = max(0.001, (pd.Timestamp(str(row.get("timestamp",""))) - pd.Timestamp(str(prev.get("timestamp","")))).total_seconds() / 3600)
                if (dist / hours) > 700 and dist > 500:
                    impossible = 1
            except Exception:
                pass

        country_dev = 1 if row.get("country","Unknown") not in user_country_history.get(uid, set()) else 0
        role_esc    = 1 if float(row.get("role_rank",0)) > float(user_role_median.get(uid,0)) else 0
        hour        = int(row.get("hour_of_day", 12))
        hour_sin    = round(_np.sin(2*_np.pi*hour/24), 4)
        hour_cos    = round(_np.cos(2*_np.pi*hour/24), 4)

        feature_rows.append([
            impossible,
            0.0,                              # hours_since_last (simplified)
            int(row.get("mfa_failed", 0)),
            int(row.get("is_known_device", 1)),
            country_dev,
            hour_sin, hour_cos,
            1 if 8 <= hour <= 18 else 0,
            int(row.get("role_rank", 0)),
            role_esc,
            int(row.get("password_reset", 0)),
            0.0,                              # peer_zscore (simplified)
        ])

        last_event[uid] = row.to_dict()

    X = _np.array(feature_rows, dtype=float)

    # ── ML scoring using saved models ─────────────────────────────────────
    iso_scores_raw   = iso_forest.decision_function(X)
    iso_scores_norm  = 100 * (1 - (iso_scores_raw - float(ISO_MIN)) / max(float(ISO_MAX) - float(ISO_MIN), 0.001))
    iso_scores_norm  = _np.clip(iso_scores_norm, 0, 100)

    xgb_proba        = xgb_model.predict_proba(X)
    xgb_pred         = xgb_model.predict(X)
    high_prob        = xgb_proba[:, 2]

    final_scores     = _np.clip(ISO_WEIGHT * iso_scores_norm + XGB_WEIGHT * (high_prob * 100), 0, 100).astype(int)
    pred_labels      = [LABEL_MAP_INV.get(int(p), "LOW") for p in xgb_pred]

    # ── Policy engine decisions ───────────────────────────────────────────
    def score_to_action(score):
        if score >= 80: return "BLOCK"
        if score >= 60: return "FORCE_MFA"
        if score >= 40: return "NOTIFY_ADMIN"
        return "ALLOW"

    # ── Build result rows ─────────────────────────────────────────────────
    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        results.append({
            "user_id":        str(row.get("user_id", "unknown")),
            "user_name":      str(row.get("user_name", row.get("user_id","Unknown"))),
            "department":     str(row.get("department", "Unknown")),
            "timestamp":      str(row.get("timestamp", "")),
            "country":        str(row.get("country","Unknown")),
            "city":           str(row.get("city","Unknown")),
            "anomaly_type":   str(row.get("anomaly_type","none")),
            "risk_label":     pred_labels[i],
            "risk_score":     int(final_scores[i]),
            "iso_score":      round(float(iso_scores_norm[i]), 1),
            "xgb_high_prob":  round(float(high_prob[i]), 4),
            "action":         score_to_action(int(final_scores[i])),
            "explanation":    [],
        })

    # ── Summary stats ─────────────────────────────────────────────────────
    high_count   = sum(1 for r in results if r["risk_label"] == "HIGH")
    med_count    = sum(1 for r in results if r["risk_label"] == "MEDIUM")
    low_count    = sum(1 for r in results if r["risk_label"] == "LOW")
    blocked      = sum(1 for r in results if r["action"] == "BLOCK")
    mfa_forced   = sum(1 for r in results if r["action"] == "FORCE_MFA")

    return {
        "source":       "uploaded_file",
        "filename":     filename,
        "total_events": len(results),
        "stats": {
            "high_risk":    high_count,
            "medium_risk":  med_count,
            "low_risk":     low_count,
            "blocked":      blocked,
            "mfa_forced":   mfa_forced,
            "total_events": len(results),
        },
        "events": results,
    }