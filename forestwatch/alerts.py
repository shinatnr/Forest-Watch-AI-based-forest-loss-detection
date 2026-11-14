"""Alert text + optional e-mail. E-mail is DISABLED unless FW_EMAIL_ENABLED=1.
Env vars: FW_SMTP_HOST, FW_SMTP_PORT (587), FW_SMTP_USER, FW_SMTP_PASS, FW_ALERT_FROM, FW_ALERT_TO."""
import os, smtplib
from email.message import EmailMessage


def alert_message(result, aoi="uploaded scene"):
    s = result["stats"]
    lines = [f"Forest Watch alert for {aoi}",
             f"Estimated forest loss: {s['hectares']:.2f} ha ({s['pct_changed']:.2f}% of valid pixels), {s['patches']} patch(es).",
             f"Alert limit: {s['alert_limit_ha']} ha."]
    if result["alert_patches"]:
        lines.append("Patches above the limit: " + ", ".join(f"#{i}" for i in result["alert_patches"]))
    if s["suspicious_patches"]:
        lines.append(f"{s['suspicious_patches']} patch(es) lie outside authorised zones - needs field verification.")
    lines.append("This system detects forest loss; it cannot prove that clearing was illegal.")
    return "\n".join(lines)


def email_enabled():
    return os.environ.get("FW_EMAIL_ENABLED") == "1"


def send_email_alert(result, aoi="uploaded scene"):
    """Returns (sent: bool, message: str). Never raises: failures are reported as text."""
    if not email_enabled():
        return False, "E-mail alerts disabled (set FW_EMAIL_ENABLED=1 to enable)."
    try:
        msg = EmailMessage(); msg["Subject"] = f"[Forest Watch] Possible deforestation: {aoi}"
        msg["From"], msg["To"] = os.environ["FW_ALERT_FROM"], os.environ["FW_ALERT_TO"]
        msg.set_content(alert_message(result, aoi))
        with smtplib.SMTP(os.environ["FW_SMTP_HOST"], int(os.environ.get("FW_SMTP_PORT", 587)), timeout=15) as s:
            s.starttls(); s.login(os.environ["FW_SMTP_USER"], os.environ["FW_SMTP_PASS"]); s.send_message(msg)
        return True, "Alert e-mail sent."
    except Exception as e:  # noqa
        return False, f"E-mail failed: {e}"
