import hashlib
import hmac
import os
import secrets
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from urllib import request as urlrequest
from urllib.error import HTTPError
from jose import jwt

ALGORITHM = "HS256"
MAGIC_LINK_TTL_MINUTES = 20
ACCESS_TOKEN_EXPIRE_DAYS = 7


def get_jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET_KEY", "")
    if not secret:
        raise RuntimeError("JWT_SECRET_KEY is missing from environment")
    return secret


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS))
    payload = {**data, "exp": expire, "iat": now}
    return jwt.encode(payload, get_jwt_secret(), algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    return jwt.decode(token, get_jwt_secret(), algorithms=[ALGORITHM])


def generate_magic_link_token() -> str:
    return secrets.token_urlsafe(48)


def hash_magic_token(raw_token: str) -> str:
    pepper = os.getenv("MAGIC_LINK_PEPPER", get_jwt_secret())
    return hmac.new(pepper.encode("utf-8"), raw_token.encode("utf-8"), hashlib.sha256).hexdigest()


def _magic_email_html(magic_link: str) -> str:
    return f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:auto;color:#111827;">
      <h2 style="margin-bottom:8px;">Sign in to CogniVault</h2>
      <p style="margin:0 0 16px;">Use the secure magic link below. This link expires in 20 minutes and can be used once.</p>
      <a href="{magic_link}" style="display:inline-block;padding:12px 20px;background:#111827;color:#ffffff;text-decoration:none;border-radius:8px;">Sign In Securely</a>
      <p style="margin-top:18px;font-size:13px;color:#6b7280;">If you did not request this, you can ignore this email.</p>
    </div>
    """


def send_magic_link_email(to_email: str, magic_link: str) -> None:
    send_email(
        to_email=to_email,
        subject="Your CogniVault Magic Link",
        html_body=_magic_email_html(magic_link),
    )


def send_email(to_email: str, subject: str, html_body: str, text_body: str = "") -> None:
    resend_api_key = os.getenv("RESEND_API_KEY", "")
    from_email = os.getenv("EMAIL_FROM", "no-reply@cognivault.ai")

    if resend_api_key:
        payload = (
            '{"from":"%s","to":["%s"],"subject":%s,"html":%s}'
            % (
                from_email,
                to_email,
                _json_escape(subject),
                _json_escape(html_body),
            )
        ).encode("utf-8")
        req = urlrequest.Request(
            "https://api.resend.com/emails",
            data=payload,
            headers={
                "Authorization": f"Bearer {resend_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=20):
                return
        except HTTPError as exc:
            raise RuntimeError(f"Resend API failed: {exc.code}") from exc

    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASS", "")
    if not (smtp_host and smtp_user and smtp_pass):
        raise RuntimeError("No email provider configured. Set RESEND_API_KEY or SMTP_* variables.")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    if text_body:
        msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(from_email, [to_email], msg.as_string())


def _json_escape(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
