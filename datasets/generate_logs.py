import csv
import random
import uuid
import os
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2

random.seed(42)

USERS = [
    {"user_id": "tanaka_h",   "name": "Tanaka Hiroshi",  "dept": "Engineering",   "home_city": "Tokyo"},
    {"user_id": "sato_m",     "name": "Sato Miyuki",     "dept": "Finance",       "home_city": "Osaka"},
    {"user_id": "admin45",    "name": "Admin User 45",   "dept": "IT Operations", "home_city": "Tokyo"},
    {"user_id": "kim_j",      "name": "Kim Junho",       "dept": "Sales",         "home_city": "Tokyo"},
    {"user_id": "yamada_r",   "name": "Yamada Riko",     "dept": "HR",            "home_city": "Tokyo"},
    {"user_id": "suzuki_t",   "name": "Suzuki Takeshi",  "dept": "Engineering",   "home_city": "Tokyo"},
    {"user_id": "chen_w",     "name": "Chen Wei",        "dept": "Engineering",   "home_city": "Tokyo"},
    {"user_id": "park_s",     "name": "Park Soyeon",     "dept": "Marketing",     "home_city": "Tokyo"},
    {"user_id": "ito_k",      "name": "Ito Kenji",       "dept": "Finance",       "home_city": "Nagoya"},
    {"user_id": "nakamura_a", "name": "Nakamura Aoi",    "dept": "DevOps",        "home_city": "Tokyo"},
]

LOCATIONS = {
    "Tokyo":     (35.6762,  139.6503, "Japan",       "103.5"),
    "Osaka":     (34.6937,  135.5023, "Japan",       "126.1"),
    "Nagoya":    (35.1815,  136.9066, "Japan",       "119.2"),
    "Moscow":    (55.7558,   37.6173, "Russia",      "77.88"),
    "Shanghai":  (31.2304,  121.4737, "China",       "101.69"),
    "New York":  (40.7128,  -74.0060, "USA",         "69.89"),
    "Seoul":     (37.5665,  126.9780, "South Korea", "175.125"),
    "London":    (51.5074,   -0.1278, "UK",          "78.32"),
    "Singapore": ( 1.3521,  103.8198, "Singapore",   "175.41"),
}

ROLES      = ["viewer", "editor", "manager", "admin", "super_admin"]
ROLE_RANK  = {r: i for i, r in enumerate(ROLES)}
DEVICES    = {u["user_id"]: [str(uuid.uuid4())[:8] for _ in range(2)] for u in USERS}

def random_ip(prefix):
    return f"{prefix}.{random.randint(1,254)}.{random.randint(1,254)}"

def make_row(user, ts, city, device, known_device, role, mfa_failed,
             mfa_success, pwd_reset, anomaly, risk, session_id):
    lat, lon, country, ip_pfx = LOCATIONS[city]
    return {
        "session_id":      session_id,
        "user_id":         user["user_id"],
        "user_name":       user["name"],
        "department":      user["dept"],
        "timestamp":       ts.isoformat(),
        "hour_of_day":     ts.hour,
        "event_type":      "login",
        "source_ip":       random_ip(ip_pfx),
        "city":            city,
        "country":         country,
        "latitude":        lat,
        "longitude":       lon,
        "device_id":       device,
        "is_known_device": known_device,
        "role_requested":  role,
        "role_rank":       ROLE_RANK[role],
        "mfa_failed":      mfa_failed,
        "mfa_success":     mfa_success,
        "password_reset":  pwd_reset,
        "anomaly_type":    anomaly,
        "risk_label":      risk,
    }

all_rows = []
base_date = datetime(2025, 1, 1)

anomaly_schedule = {
    "tanaka_h":  "impossible_travel",
    "admin45":   "mfa_brute_force",
    "sato_m":    "new_country",
    "kim_j":     "password_reset_flood",
    "yamada_r":  "privilege_escalation",
}

for day_offset in range(30):
    current = base_date + timedelta(days=day_offset)

    for user in USERS:
        uid = user["user_id"]

        # Normal logins (1-3 per day)
        for _ in range(random.randint(1, 3)):
            sid  = str(uuid.uuid4())[:8]
            hour = random.randint(8, 18)
            ts   = current.replace(hour=hour, minute=random.randint(0, 59), second=0)
            dev  = random.choice(DEVICES[uid])
            role = random.choice(ROLES[:2])
            all_rows.append(make_row(user, ts, user["home_city"], dev, True,
                                     role, 0, 1, 0, "none", "LOW", sid))

        # Inject anomaly every 5 days
        if uid in anomaly_schedule and day_offset % 5 == 0:
            sid   = str(uuid.uuid4())[:8]
            atype = anomaly_schedule[uid]

            if atype == "impossible_travel":
                # Normal Tokyo login
                ts1  = current.replace(hour=8, minute=30, second=0)
                dev1 = random.choice(DEVICES[uid])
                all_rows.append(make_row(user, ts1, "Tokyo", dev1, True,
                                         "viewer", 0, 1, 0, "none", "LOW", sid))
                # Moscow login 2 hours later
                ts2  = current.replace(hour=10, minute=45, second=0)
                dev2 = str(uuid.uuid4())[:8]
                all_rows.append(make_row(user, ts2, "Moscow", dev2, False,
                                         "admin", 0, 1, 0, "impossible_travel", "HIGH",
                                         str(uuid.uuid4())[:8]))

            elif atype == "mfa_brute_force":
                for i in range(5):
                    ts_i = current.replace(hour=22, minute=i*2, second=0)
                    dev  = str(uuid.uuid4())[:8]
                    risk = "HIGH" if i >= 2 else "MEDIUM"
                    all_rows.append(make_row(user, ts_i, "Tokyo", dev, False,
                                             "admin", i+1, 0, 0, "mfa_brute_force",
                                             risk, str(uuid.uuid4())[:8]))

            elif atype == "new_country":
                foreign = random.choice(["Seoul", "London", "Singapore", "New York"])
                ts  = current.replace(hour=random.randint(1, 5), minute=0, second=0)
                dev = str(uuid.uuid4())[:8]
                all_rows.append(make_row(user, ts, foreign, dev, False,
                                         "manager", 0, 1, 0, "new_country_login",
                                         "MEDIUM", sid))

            elif atype == "password_reset_flood":
                for i in range(3):
                    ts_i = current.replace(hour=2, minute=i*15, second=0)
                    dev  = random.choice(DEVICES[uid])
                    all_rows.append(make_row(user, ts_i, user["home_city"], dev, True,
                                             "viewer", 0, 1, 1, "password_reset_flood",
                                             "MEDIUM", str(uuid.uuid4())[:8]))

            elif atype == "privilege_escalation":
                ts  = current.replace(hour=random.randint(9, 17), minute=0, second=0)
                dev = random.choice(DEVICES[uid])
                all_rows.append(make_row(user, ts, user["home_city"], dev, True,
                                         "super_admin", 0, 1, 0, "privilege_escalation",
                                         "HIGH", sid))

# Sort and write
all_rows.sort(key=lambda r: r["timestamp"])
os.makedirs("datasets", exist_ok=True)

fieldnames = list(all_rows[0].keys())
with open("datasets/iam_logs.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

from collections import Counter
labels    = Counter(r["risk_label"]  for r in all_rows)
anomalies = Counter(r["anomaly_type"] for r in all_rows if r["anomaly_type"] != "none")

print(f"Generated {len(all_rows)} log entries → datasets/iam_logs.csv")
print(f"Risk label breakdown : {dict(labels)}")
print(f"Anomaly types injected: {dict(anomalies)}")