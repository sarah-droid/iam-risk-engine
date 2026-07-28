import pandas as pd
import json
import os
import sqlite3
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# POLICY ENGINE
# Converts a 0-100 risk score into a concrete action
# Thresholds:
#   80-100  → BLOCK         (アクセス拒否)
#   60-79   → FORCE_MFA     (多要素認証を強制)
#   40-59   → NOTIFY_ADMIN  (管理者に通知)
#   0-39    → ALLOW         (アクセスを許可)
# ═══════════════════════════════════════════════════════════════════════════════

POLICY_RULES = [
    {"min_score": 80, "max_score": 100, "action": "BLOCK",        "action_ja": "アクセス拒否",     "severity": "CRITICAL"},
    {"min_score": 60, "max_score": 79,  "action": "FORCE_MFA",    "action_ja": "多要素認証を強制", "severity": "HIGH"},
    {"min_score": 40, "max_score": 59,  "action": "NOTIFY_ADMIN", "action_ja": "管理者に通知",     "severity": "MEDIUM"},
    {"min_score": 0,  "max_score": 39,  "action": "ALLOW",        "action_ja": "アクセスを許可",   "severity": "LOW"},
]

def get_action(risk_score):
    """Return policy decision for a given risk score."""
    for rule in POLICY_RULES:
        if rule["min_score"] <= risk_score <= rule["max_score"]:
            return {
                "action":      rule["action"],
                "action_ja":   rule["action_ja"],
                "severity":    rule["severity"],
                "risk_score":  risk_score,
            }
    return {"action": "ALLOW", "action_ja": "アクセスを許可", "severity": "LOW", "risk_score": risk_score}


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT DATABASE
# Every action taken is stored in SQLite for compliance + auditability
# Japanese enterprise requirement: 監査ログ
# ═══════════════════════════════════════════════════════════════════════════════

os.makedirs("datasets", exist_ok=True)
DB_PATH = "datasets/audit.db"

conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_log (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp     TEXT NOT NULL,
        session_id    TEXT,
        user_id       TEXT NOT NULL,
        user_name     TEXT,
        department    TEXT,
        country       TEXT,
        anomaly_type  TEXT,
        risk_score    INTEGER NOT NULL,
        risk_label    TEXT,
        action_taken  TEXT NOT NULL,
        action_ja     TEXT,
        severity      TEXT,
        top_reasons   TEXT,
        triggered_by  TEXT DEFAULT 'AUTO_POLICY_ENGINE'
    )
""")
conn.commit()
print(f"✅ Audit database ready → {DB_PATH}")


def log_action(session_id, user_id, user_name, department, country,
               anomaly_type, risk_score, risk_label, action, action_ja,
               severity, top_reasons, triggered_by="AUTO_POLICY_ENGINE"):
    """Write one audit record to SQLite."""
    cursor.execute("""
        INSERT INTO audit_log
        (timestamp, session_id, user_id, user_name, department, country,
         anomaly_type, risk_score, risk_label, action_taken, action_ja,
         severity, top_reasons, triggered_by)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.utcnow().isoformat(),
        session_id, user_id, user_name, department, country,
        anomaly_type, risk_score, risk_label, action, action_ja,
        severity, top_reasons, triggered_by
    ))
    conn.commit()


# ═══════════════════════════════════════════════════════════════════════════════
# REMEDIATION ACTIONS
# In production these call real APIs (Okta, Azure AD, Slack, Jira)
# Here we simulate them with console output + audit logging
# ═══════════════════════════════════════════════════════════════════════════════

def action_block(user_id, user_name, risk_score):
    print(f"  🚫 BLOCK        | {user_name} ({user_id}) | Score: {risk_score}")
    print(f"     → Disabling account in IAM provider ...")
    print(f"     → Terminating all active sessions ...")
    print(f"     → Sending alert to SOC team ...")

def action_force_mfa(user_id, user_name, risk_score):
    print(f"  🔐 FORCE MFA    | {user_name} ({user_id}) | Score: {risk_score}")
    print(f"     → Invalidating current session token ...")
    print(f"     → Sending MFA challenge via Authenticator app ...")

def action_notify_admin(user_id, user_name, risk_score):
    print(f"  📧 NOTIFY ADMIN | {user_name} ({user_id}) | Score: {risk_score}")
    print(f"     → Posting alert to #security-alerts Slack channel ...")
    print(f"     → Creating Jira ticket for review ...")

def action_allow(user_id, user_name, risk_score):
    print(f"  ✅ ALLOW        | {user_name} ({user_id}) | Score: {risk_score}")

REMEDIATION_MAP = {
    "BLOCK":        action_block,
    "FORCE_MFA":    action_force_mfa,
    "NOTIFY_ADMIN": action_notify_admin,
    "ALLOW":        action_allow,
}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — Apply policy to all scored events
# ═══════════════════════════════════════════════════════════════════════════════

scored_df = pd.read_csv("datasets/scored_events.csv")
print(f"\nLoaded {len(scored_df)} scored events")

# Add action columns
actions_taken   = []
actions_ja      = []
severities      = []

print("\n" + "=" * 60)
print("Applying Policy Engine to all events ...")
print("=" * 60)

block_count  = 0
mfa_count    = 0
notify_count = 0
allow_count  = 0

for _, row in scored_df.iterrows():
    decision = get_action(int(row["final_risk_score"]))

    actions_taken.append(decision["action"])
    actions_ja.append(decision["action_ja"])
    severities.append(decision["severity"])

    # Run remediation for non-ALLOW actions only (skip printing ALLOW to keep output clean)
    if decision["action"] != "ALLOW":
        REMEDIATION_MAP[decision["action"]](
            row["user_id"], row["user_name"], row["final_risk_score"]
        )

    # Count
    if decision["action"] == "BLOCK":        block_count  += 1
    elif decision["action"] == "FORCE_MFA":  mfa_count    += 1
    elif decision["action"] == "NOTIFY_ADMIN": notify_count += 1
    else:                                    allow_count  += 1

    # Write to audit log
    log_action(
        session_id   = str(row.get("session_id", "")),
        user_id      = row["user_id"],
        user_name    = row["user_name"],
        department   = row["department"],
        country      = row["country"],
        anomaly_type = row["anomaly_type"],
        risk_score   = int(row["final_risk_score"]),
        risk_label   = row["risk_label"],
        action       = decision["action"],
        action_ja    = decision["action_ja"],
        severity     = decision["severity"],
        top_reasons  = str(row.get("shap_explanation", "")),
    )

# Attach decisions to dataframe
scored_df["action_taken"] = actions_taken
scored_df["action_ja"]    = actions_ja
scored_df["severity"]     = severities

# Save enriched dataset
scored_df.to_csv("datasets/scored_events.csv", index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Policy Engine Summary · ポリシーエンジン結果")
print("=" * 60)
print(f"  🚫 BLOCK        (アクセス拒否)     : {block_count}")
print(f"  🔐 FORCE_MFA    (MFA強制)          : {mfa_count}")
print(f"  📧 NOTIFY_ADMIN (管理者通知)       : {notify_count}")
print(f"  ✅ ALLOW        (許可)             : {allow_count}")
print(f"  ─────────────────────────────────────")
print(f"  Total events processed             : {len(scored_df)}")

# ── Audit log preview ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Recent Audit Log entries · 監査ログ")
print("=" * 60)
audit_df = pd.read_sql("SELECT * FROM audit_log ORDER BY id DESC LIMIT 8", conn)
for _, row in audit_df.iterrows():
    print(f"  [{row['timestamp'][:19]}] {row['user_name']:20s} | "
          f"Score: {row['risk_score']:3d} | "
          f"{row['action_taken']:12s} | "
          f"{row['anomaly_type']}")

conn.close()
print(f"\n✅ Audit log saved → {DB_PATH}")
print(f"✅ scored_events.csv updated with actions → datasets/scored_events.csv")