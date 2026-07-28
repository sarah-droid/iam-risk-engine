"""
Email Alert Module
===================
Sends real email alerts via Gmail SMTP when high-risk IAM events occur,
and manages the list of admin email addresses who should receive them.

Setup required (one-time):
  1. Enable 2-Step Verification on the sending Gmail account
  2. Generate an App Password: https://myaccount.google.com/apppasswords
  3. Set environment variables before starting the API:
       SMTP_EMAIL=youraccount@gmail.com
       SMTP_APP_PASSWORD=xxxxxxxxxxxxxxxx   (16-character app password, no spaces)

  On Windows (Git Bash), set these in the same terminal before running uvicorn:
       export SMTP_EMAIL="youraccount@gmail.com"
       export SMTP_APP_PASSWORD="xxxxxxxxxxxxxxxx"
       python -m uvicorn api.main:app --reload
"""

import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465

ADMIN_EMAILS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "admin_emails.json"
)


# ── Admin email list management ──────────────────────────────────────────

def load_admin_emails():
    """Returns the current list of admin emails. Creates the file with
    an empty list if it doesn't exist yet."""
    if not os.path.exists(ADMIN_EMAILS_PATH):
        os.makedirs(os.path.dirname(ADMIN_EMAILS_PATH), exist_ok=True)
        with open(ADMIN_EMAILS_PATH, "w") as f:
            json.dump({"admins": []}, f, indent=2)
        return []

    with open(ADMIN_EMAILS_PATH) as f:
        data = json.load(f)
    return data.get("admins", [])


def save_admin_emails(emails):
    """Overwrites the admin email list with the given list of dicts:
    [{"name": "...", "email": "..."}]"""
    os.makedirs(os.path.dirname(ADMIN_EMAILS_PATH), exist_ok=True)
    with open(ADMIN_EMAILS_PATH, "w") as f:
        json.dump({"admins": emails}, f, indent=2)


def add_admin_email(name, email):
    admins = load_admin_emails()
    # Avoid duplicate entries
    if any(a["email"].lower() == email.lower() for a in admins):
        return admins, False  # already exists
    admins.append({"name": name, "email": email})
    save_admin_emails(admins)
    return admins, True


def remove_admin_email(email):
    admins = load_admin_emails()
    new_admins = [a for a in admins if a["email"].lower() != email.lower()]
    removed = len(new_admins) != len(admins)
    save_admin_emails(new_admins)
    return new_admins, removed


# ── Email sending ─────────────────────────────────────────────────────────

def build_alert_email_html(event_data):
    """Builds an HTML email body for a security alert."""
    user_name = event_data.get("user_name", "Unknown User")
    user_id = event_data.get("user_id", "")
    risk_score = event_data.get("risk_score", 0)
    risk_label = event_data.get("risk_label", "MEDIUM")
    action = event_data.get("action", "NOTIFY_ADMIN")
    anomaly_type = event_data.get("anomaly_type", "manual_alert")
    country = event_data.get("country", "Unknown")
    timestamp = event_data.get("timestamp", datetime.utcnow().isoformat())
    reasons = event_data.get("reasons", [])

    color = "#ef4444" if risk_label == "HIGH" else "#f59e0b" if risk_label == "MEDIUM" else "#10b981"

    reasons_html = "".join(f"<li style='margin-bottom:4px'>{r}</li>" for r in reasons) if reasons else "<li>Manually triggered by admin</li>"

    html = f"""
    <html>
    <body style="font-family: -apple-system, Segoe UI, sans-serif; background:#f4f6fb; padding:24px;">
      <div style="max-width:520px; margin:0 auto; background:#ffffff; border-radius:12px; overflow:hidden; border:1px solid #e2e8f0;">

        <div style="background:#0a0e1a; padding:20px 24px;">
          <span style="color:#fff; font-size:16px; font-weight:700;">🛡 IdentityGuard AI</span>
          <div style="color:#94a3b8; font-size:11px; margin-top:2px;">IAM Threat Detection Alert</div>
        </div>

        <div style="padding:24px;">
          <div style="display:inline-block; background:{color}1a; color:{color}; border:1px solid {color}4d; border-radius:20px; padding:4px 14px; font-size:13px; font-weight:700; margin-bottom:16px;">
            {risk_label} RISK — Score {risk_score}/100
          </div>

          <h2 style="margin:0 0 4px; font-size:18px; color:#1e2433;">Security Alert: {user_name}</h2>
          <p style="margin:0 0 16px; color:#64748b; font-size:13px;">{user_id} · {country} · {timestamp[:19].replace('T',' ')} UTC</p>

          <table style="width:100%; border-collapse:collapse; margin-bottom:16px; font-size:13px;">
            <tr><td style="padding:6px 0; color:#64748b;">Anomaly Type</td><td style="padding:6px 0; text-align:right; font-weight:600; color:#1e2433;">{anomaly_type.replace('_',' ')}</td></tr>
            <tr><td style="padding:6px 0; color:#64748b; border-top:1px solid #f1f5f9;">Recommended Action</td><td style="padding:6px 0; text-align:right; font-weight:600; color:#1e2433; border-top:1px solid #f1f5f9;">{action.replace('_',' ')}</td></tr>
          </table>

          <div style="background:#f8fafc; border-radius:8px; padding:14px 16px; margin-bottom:20px;">
            <div style="font-size:12px; font-weight:700; color:#475569; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.5px;">Why this was flagged</div>
            <ul style="margin:0; padding-left:18px; color:#334155; font-size:13px;">
              {reasons_html}
            </ul>
          </div>

          <p style="font-size:11px; color:#94a3b8; margin:0;">
            This is an automated alert from your local IdentityGuard AI instance.
            Log in to the dashboard to review the full event and take action.
          </p>
        </div>
      </div>
    </body>
    </html>
    """
    return html


def send_alert_email(event_data, recipient_emails=None):
    """
    Sends a security alert email to all admins (or a specific list if given).
    Returns (success: bool, message: str, sent_to: list).
    """
    sender_email = os.environ.get("SMTP_EMAIL")
    sender_password = os.environ.get("SMTP_APP_PASSWORD")

    if not sender_email or not sender_password:
        return False, "SMTP_EMAIL and SMTP_APP_PASSWORD environment variables are not set.", []

    if recipient_emails is None:
        admins = load_admin_emails()
        recipient_emails = [a["email"] for a in admins]

    if not recipient_emails:
        return False, "No admin email addresses configured. Add one in Settings first.", []

    user_name = event_data.get("user_name", "Unknown User")
    risk_label = event_data.get("risk_label", "MEDIUM")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🚨 [{risk_label} RISK] IAM Alert — {user_name}"
    msg["From"] = sender_email
    msg["To"] = ", ".join(recipient_emails)

    html_body = build_alert_email_html(event_data)
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_emails, msg.as_string())
        return True, f"Alert email sent to {len(recipient_emails)} admin(s).", recipient_emails

    except smtplib.SMTPAuthenticationError:
        return False, "SMTP authentication failed. Check SMTP_EMAIL and SMTP_APP_PASSWORD (must be a Gmail App Password, not your regular password).", []
    except Exception as e:
        return False, f"Failed to send email: {str(e)}", []


if __name__ == "__main__":
    # Quick manual test — run with: python api/email_alerts.py
    print("Loaded admin emails:", load_admin_emails())

    test_event = {
        "user_name": "Tanaka Hiroshi",
        "user_id": "tanaka_h",
        "risk_score": 94,
        "risk_label": "HIGH",
        "action": "BLOCK",
        "anomaly_type": "impossible_travel",
        "country": "Russia",
        "timestamp": datetime.utcnow().isoformat(),
        "reasons": [
            "Impossible travel: Moscow→Tokyo in 2 hours",
            "Privileged role requested",
            "New device fingerprint",
        ],
    }

    success, message, sent_to = send_alert_email(test_event)
    print(f"Success: {success}")
    print(f"Message: {message}")
    print(f"Sent to: {sent_to}")