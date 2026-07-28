import pandas as pd
import numpy as np
import os
from math import radians, sin, cos, sqrt, atan2

# ── Load raw logs ─────────────────────────────────────────────────────────────
df = pd.read_csv("datasets/iam_logs.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"Loaded {len(df)} raw log rows")

# ── Build historical context per user ────────────────────────────────────────
user_country_history = df.groupby("user_id")["country"].apply(set).to_dict()
user_role_median     = df.groupby("user_id")["role_rank"].median().to_dict()

# ── Department peer daily event counts ───────────────────────────────────────
df["date"] = df["timestamp"].dt.date

daily_counts = df.groupby(["user_id", "date"]).size().reset_index(name="daily_count")
dept_map     = df[["user_id","department"]].drop_duplicates().set_index("user_id")["department"].to_dict()
daily_counts["department"] = daily_counts["user_id"].map(dept_map)

dept_stats   = daily_counts.groupby("department")["daily_count"].agg(["mean","std"]).to_dict()

# ── Rolling 30-min MFA failure count ─────────────────────────────────────────
df["mfa_fail_30min"] = 0
for uid, group in df.groupby("user_id"):
    idxs = group.index.tolist()
    for i, idx in enumerate(idxs):
        window_start = df.at[idx, "timestamp"] - pd.Timedelta(minutes=30)
        fails = sum(
            1 for j in idxs[:i+1]
            if df.at[j, "timestamp"] >= window_start and df.at[j, "mfa_failed"] > 0
        )
        df.at[idx, "mfa_fail_30min"] = fails

# ── Rolling 7-day password reset count ───────────────────────────────────────
df["pwd_reset_7d"] = 0
for uid, group in df.groupby("user_id"):
    idxs = group.index.tolist()
    for i, idx in enumerate(idxs):
        window_start = df.at[idx, "timestamp"] - pd.Timedelta(days=7)
        resets = sum(
            1 for j in idxs[:i+1]
            if df.at[j, "timestamp"] >= window_start and df.at[j, "password_reset"] > 0
        )
        df.at[idx, "pwd_reset_7d"] = resets

# ── Haversine distance helper ─────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a    = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# ── Row-by-row feature computation ───────────────────────────────────────────
features   = []
last_event = {}   # user_id -> previous row

for _, row in df.iterrows():
    uid  = row["user_id"]
    prev = last_event.get(uid)

    # -- Impossible travel ----------------------------------------------------
    impossible = 0
    if prev is not None:
        dist_km = haversine_km(prev["latitude"], prev["longitude"],
                               row["latitude"],  row["longitude"])
        hours   = max((row["timestamp"] - prev["timestamp"]).total_seconds() / 3600, 0.001)
        speed   = dist_km / hours
        if speed > 700 and dist_km > 500:
            impossible = 1

    # -- Time since last login ------------------------------------------------
    hours_gap = 0.0
    if prev is not None:
        hours_gap = (row["timestamp"] - prev["timestamp"]).total_seconds() / 3600

    # -- Country deviation ----------------------------------------------------
    hist_countries = user_country_history.get(uid, set())
    country_dev    = 1 if row["country"] not in hist_countries else 0

    # -- Role escalation ------------------------------------------------------
    median_rank = user_role_median.get(uid, 0)
    role_esc    = 1 if row["role_rank"] > median_rank else 0

    # -- Cyclical hour encoding -----------------------------------------------
    hour     = row["hour_of_day"]
    hour_sin = round(np.sin(2 * np.pi * hour / 24), 4)
    hour_cos = round(np.cos(2 * np.pi * hour / 24), 4)

    # -- Business hours flag --------------------------------------------------
    is_business_hours = 1 if 8 <= hour <= 18 else 0

    # -- Peer z-score ---------------------------------------------------------
    dept      = dept_map.get(uid, "Unknown")
    day_count = daily_counts.loc[
        (daily_counts["user_id"] == uid) & (daily_counts["date"] == row["date"]),
        "daily_count"
    ]
    if not day_count.empty:
        mu        = dept_stats["mean"].get(dept, 1)
        sigma     = dept_stats["std"].get(dept, 1) or 1
        peer_z    = round((day_count.values[0] - mu) / sigma, 3)
    else:
        peer_z    = 0.0

    features.append({
        # Identifiers
        "session_id":        row["session_id"],
        "user_id":           uid,
        "user_name":         row["user_name"],
        "department":        row["department"],
        "timestamp":         row["timestamp"].isoformat(),
        "event_type":        row["event_type"],
        "country":           row["country"],
        "city":              row["city"],

        # ML Features
        "impossible_travel":  impossible,
        "hours_since_last":   round(hours_gap, 2),
        "mfa_fail_count":     int(row["mfa_fail_30min"]),
        "is_known_device":    int(row["is_known_device"]),
        "country_deviation":  country_dev,
        "hour_sin":           hour_sin,
        "hour_cos":           hour_cos,
        "is_business_hours":  is_business_hours,
        "role_rank":          int(row["role_rank"]),
        "role_escalation":    role_esc,
        "pwd_reset_7d":       int(row["pwd_reset_7d"]),
        "peer_zscore":        peer_z,

        # Labels
        "risk_label":         row["risk_label"],
        "anomaly_type":       row["anomaly_type"],
    })

    last_event[uid] = row

# ── Save features ─────────────────────────────────────────────────────────────
feat_df = pd.DataFrame(features)
os.makedirs("datasets", exist_ok=True)
feat_df.to_csv("datasets/features.csv", index=False)

print(f"Saved {len(feat_df)} feature rows → datasets/features.csv")
print(f"\nLabel distribution:\n{feat_df['risk_label'].value_counts().to_string()}")
print(f"\nSample HIGH risk rows:")
print(feat_df[feat_df["risk_label"] == "HIGH"][
    ["user_name", "anomaly_type", "impossible_travel",
     "mfa_fail_count", "country_deviation", "role_escalation", "peer_zscore"]
].head(6).to_string())