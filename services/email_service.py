import os
import smtplib
from email.mime.text import MIMEText

SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_EMAIL = os.environ.get("SMTP_EMAIL", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173")


def send_reset_email(to_email: str, token: str):
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        return

    reset_link = f"{FRONTEND_URL}/reset-password?token={token}"
    subject = "DocuVerse — Password Reset"
    body = f"""
    <html>
      <body style="font-family:sans-serif;background:#f8fafc;padding:24px">
        <div style="max-width:480px;margin:auto;background:white;border-radius:12px;padding:32px">
          <h2 style="margin-top:0">DocuVerse</h2>
          <p>You requested a password reset. Click the button below:</p>
          <a href="{reset_link}"
             style="display:inline-block;padding:12px 24px;background:#6366f1;color:white;text-decoration:none;border-radius:8px">
             Reset Password
          </a>
          <p style="margin-top:24px;color:#64748b;font-size:13px">
            This link expires in 1 hour. If you didn't request this, ignore this email.
          </p>
        </div>
      </body>
    </html>
    """

    msg = MIMEText(body, "html")
    msg["Subject"] = subject
    msg["From"] = SMTP_EMAIL
    msg["To"] = to_email

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as server:
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.send_message(msg)
