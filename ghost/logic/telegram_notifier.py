# logic/telegram_notifier.py
import os
import io
import threading
import requests
import config

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"

def _get_creds():
    token = getattr(config, "TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = getattr(config, "TELEGRAM_CHAT_ID", "").strip()
    enabled = getattr(config, "TELEGRAM_ENABLED", False)
    return token, chat_id, bool(enabled)


def send_message(text: str, parse_mode: str = "HTML") -> bool:
    """Send a plain text message. Non-blocking (fires in background thread)."""
    token, chat_id, enabled = _get_creds()
    if not enabled or not token or not chat_id:
        return False

    def _send():
        try:
            url = TELEGRAM_API.format(token=token, method="sendMessage")
            resp = requests.post(url, json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode
            }, timeout=10)
            if not resp.ok:
                print(f"[TELEGRAM] sendMessage failed: {resp.text}")
        except Exception as e:
            print(f"[TELEGRAM] sendMessage error: {e}")

    threading.Thread(target=_send, daemon=True).start()
    return True


def send_photo(image_bytes: bytes, caption: str = "") -> bool:
    """Send a JPEG screenshot with caption. Non-blocking."""
    token, chat_id, enabled = _get_creds()
    if not enabled or not token or not chat_id:
        return False

    def _send():
        try:
            url = TELEGRAM_API.format(token=token, method="sendPhoto")
            files = {"photo": ("snapshot.jpg", io.BytesIO(image_bytes), "image/jpeg")}
            data = {"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"}
            resp = requests.post(url, data=data, files=files, timeout=15)
            if not resp.ok:
                print(f"[TELEGRAM] sendPhoto failed: {resp.text}")
        except Exception as e:
            print(f"[TELEGRAM] sendPhoto error: {e}")

    threading.Thread(target=_send, daemon=True).start()
    return True


def send_alert_with_snapshot(frame_bgr, room: str, zone: str, duration: float, money: float) -> bool:
    """
    Encode a CV2 frame as JPEG and send it as a Telegram photo alert.
    Call this from detector.py when a waste event fires.
    """
    try:
        import cv2
        import numpy as np

        if frame_bgr is None or getattr(frame_bgr, 'size', 0) == 0:
            return False

        # Encode frame to JPEG bytes
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return False
        image_bytes = buf.tobytes()

        caption = (
            f"⚠️ <b>Energy Waste Alert</b>\n"
            f"🏢 <b>Room:</b> {room}\n"
            f"📍 <b>Zone:</b> {zone}\n"
            f"⏱ <b>Duration:</b> {round(duration, 1)}s\n"
            f"💸 <b>Cost Wasted:</b> ₹{round(money, 2)}\n"
            f"🕐 <b>Time:</b> {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        return send_photo(image_bytes, caption)
    except Exception as e:
        print(f"[TELEGRAM] send_alert_with_snapshot error: {e}")
        return False


def send_test_message() -> tuple:
    """Send a test message. Returns (success, message) tuple."""
    token, chat_id, enabled = _get_creds()
    if not token or not chat_id:
        return False, "Bot token or Chat ID not configured."
    try:
        url = TELEGRAM_API.format(token=token, method="sendMessage")
        resp = requests.post(url, json={
            "chat_id": chat_id,
            "text": "✅ <b>VisionCore</b> — Telegram channel verified successfully!",
            "parse_mode": "HTML"
        }, timeout=10)
        if resp.ok:
            return True, "Test message sent to Telegram!"
        return False, f"Telegram API error: {resp.text}"
    except Exception as e:
        return False, f"Connection error: {str(e)}"
