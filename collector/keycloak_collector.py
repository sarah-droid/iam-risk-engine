import requests
import csv
import os
import time
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# Update these to match your local Keycloak setup
# ═══════════════════════════════════════════════════════════════════════════════

KEYCLOAK_URL   = "http://localhost:8080"
ADMIN_USER     = "admin"
ADMIN_PASSWORD = "admin"
REALM          = "master"          # change to your test realm name if you made one
OUTPUT_CSV     = "datasets/keycloak_logs.csv"
POLL_INTERVAL  = 10                 # seconds between polls when running continuously
MAX_EVENTS     = 100                # events per page

# Department/location enrichment lookup — Keycloak doesn't know this,
# so we attach it ourselves based on username, same as our simulated dataset
USER_DEPT_MAP = {
    "tanaka_h":   {"name": "Tanaka Hiroshi",  "dept": "Engineering"},
    "sato_m":     {"name": "Sato Miyuki",     "dept": "Finance"},
    "admin45":    {"name": "Admin User 45",   "dept": "IT Operations"},
    "kim_j":      {"name": "Kim Junho",       "dept": "Sales"},
    "yamada_r":   {"name": "Yamada Riko",     "dept": "HR"},
}
DEFAULT_DEPT = {"name": "Unknown User", "dept": "Unknown"}

# Keycloak event types we care about, mapped to our anomaly vocabulary
EVENT_TYPE_MAP = {
    "LOGIN":            "none",
    "LOGIN_ERROR":      "failed_login",
    "LOGOUT":            "none",
    "REGISTER":          "none",
    "UPDATE_PASSWORD":   "password_reset",
    "RESET_PASSWORD":    "password_reset",
    "RESET_PASSWORD_ERROR": "password_reset_flood",
    "CLIENT_LOGIN":      "none",
    "CODE_TO_TOKEN":     "none",
    "REFRESH_TOKEN":     "none",
}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Authenticate and get an admin access token
# ═══════════════════════════════════════════════════════════════════════════════

def get_admin_token():
    url = f"{KEYCLOAK_URL}/realms/master/protocol/openid-connect/token"
    data = {
        "client_id":  "admin-cli",
        "username":   ADMIN_USER,
        "password":   ADMIN_PASSWORD,
        "grant_type": "password",
    }
    resp = requests.post(url, data=data, timeout=10)
    resp.raise_for_status()
    return resp.json()["access_token"]


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Fetch user events from the realm
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_events(token, first=0, max_results=MAX_EVENTS):
    url = f"{KEYCLOAK_URL}/admin/realms/{REALM}/events"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"first": first, "max": max_results}
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Convert a raw Keycloak event into our feature-engineering CSV schema
# Matches the columns produced by generate_logs.py so the SAME
# extract_features.py script can process either dataset unchanged.
# ═══════════════════════════════════════════════════════════════════════════════

def transform_event(event, mfa_fail_tracker):
    username  = event.get("details", {}).get("username") or event.get("userId", "unknown")
    user_info = USER_DEPT_MAP.get(username, DEFAULT_DEPT)

    raw_type     = event.get("type", "LOGIN")
    anomaly_type = EVENT_TYPE_MAP.get(raw_type, "none")

    ts_millis = event.get("time", int(time.time() * 1000))
    ts        = datetime.fromtimestamp(ts_millis / 1000)

    ip_address = event.get("ipAddress", "127.0.0.1")

    # Track consecutive login errors per user as a simple MFA-fail proxy
    if raw_type == "LOGIN_ERROR":
        mfa_fail_tracker[username] = mfa_fail_tracker.get(username, 0) + 1
    else:
        mfa_fail_tracker[username] = 0
    mfa_failed  = 1 if raw_type == "LOGIN_ERROR" else 0
    mfa_success = 1 if raw_type == "LOGIN" else 0

    risk_label = "LOW"
    if anomaly_type in ("password_reset_flood",) or mfa_fail_tracker.get(username, 0) >= 3:
        risk_label = "HIGH"
    elif anomaly_type in ("failed_login", "password_reset"):
        risk_label = "MEDIUM"

    return {
        "session_id":      event.get("sessionId", "")[:8] or str(int(time.time())),
        "user_id":         username,
        "user_name":       user_info["name"],
        "department":      user_info["dept"],
        "timestamp":       ts.isoformat(),
        "hour_of_day":      ts.hour,
        "event_type":      "login" if "LOGIN" in raw_type else raw_type.lower(),
        "source_ip":       ip_address,
        "city":            "Unknown",          # Keycloak doesn't geolocate IPs natively
        "country":         "Unknown",
        "latitude":        0.0,
        "longitude":       0.0,
        "device_id":       event.get("details", {}).get("device_id", "unknown_device"),
        "is_known_device": True,                # default; refine later with real device tracking
        "role_requested":  "viewer",             # Keycloak events don't carry role info directly
        "role_rank":       0,
        "mfa_failed":      mfa_failed,
        "mfa_success":     mfa_success,
        "password_reset":  1 if anomaly_type == "password_reset" else 0,
        "anomaly_type":    anomaly_type,
        "risk_label":      risk_label,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Write events to CSV (append mode, creates file with header if new)
# ═══════════════════════════════════════════════════════════════════════════════

FIELDNAMES = [
    "session_id","user_id","user_name","department","timestamp","hour_of_day",
    "event_type","source_ip","city","country","latitude","longitude",
    "device_id","is_known_device","role_requested","role_rank",
    "mfa_failed","mfa_success","password_reset","anomaly_type","risk_label",
]

def write_events_to_csv(events, path=OUTPUT_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.isfile(path)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for row in events:
            writer.writerow(row)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Run once (pull whatever events exist right now)
# ═══════════════════════════════════════════════════════════════════════════════

def run_once():
    print(f"Connecting to Keycloak at {KEYCLOAK_URL} ...")
    token = get_admin_token()
    print("✅ Authenticated successfully")

    raw_events = fetch_events(token)
    print(f"Fetched {len(raw_events)} raw events from realm '{REALM}'")

    if not raw_events:
        print("No events found. Make sure event logging is enabled:")
        print("  Keycloak Admin Console → Realm Settings → Events → Save Events: ON")
        return

    mfa_fail_tracker = {}
    transformed = [transform_event(e, mfa_fail_tracker) for e in raw_events]

    write_events_to_csv(transformed)
    print(f"✅ Wrote {len(transformed)} events → {OUTPUT_CSV}")

    # Quick summary
    from collections import Counter
    risk_counts    = Counter(e["risk_label"] for e in transformed)
    anomaly_counts = Counter(e["anomaly_type"] for e in transformed if e["anomaly_type"] != "none")
    print(f"\nRisk breakdown this batch : {dict(risk_counts)}")
    print(f"Anomaly types this batch  : {dict(anomaly_counts)}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Run continuously (polls every N seconds, good for a live demo)
# ═══════════════════════════════════════════════════════════════════════════════

def run_continuous():
    print(f"Starting continuous collector — polling every {POLL_INTERVAL}s. Ctrl+C to stop.\n")
    token = get_admin_token()
    token_fetched_at = time.time()
    mfa_fail_tracker = {}
    seen_event_ids = set()

    while True:
        # Refresh token every 4 minutes (Keycloak default access tokens expire at 5 min)
        if time.time() - token_fetched_at > 240:
            token = get_admin_token()
            token_fetched_at = time.time()

        try:
            raw_events = fetch_events(token)
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Request failed: {e}")
            time.sleep(POLL_INTERVAL)
            continue

        new_events = [e for e in raw_events if e.get("time") not in seen_event_ids]
        for e in new_events:
            seen_event_ids.add(e.get("time"))

        if new_events:
            transformed = [transform_event(e, mfa_fail_tracker) for e in new_events]
            write_events_to_csv(transformed)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] +{len(transformed)} new events → {OUTPUT_CSV}")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] No new events.")

        time.sleep(POLL_INTERVAL)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--watch":
        run_continuous()
    else:
        run_once()