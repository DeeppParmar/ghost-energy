import os

# config.py - Monitoring Settings

ROOM_NAME = "Advanced Physics Lab"
CAMERA_SOURCE = 0 # Use 0 for Webcam, or "rtsp://..." for CCTV
ALERT_DELAY_SECONDS = 30 # Time empty before alert

# ── Energy Cost Model Constants ──
# Used for computing Money_Wasted in energy audit CSV.
BULB_WATTAGE = 200          # Watts (room lighting power baseline)
ELECTRICITY_RATE = 8.0    # ₹ per kWh

# Light Classification 
BRIGHTNESS_THRESHOLD = 40 # Below this = Dark
WINDOW_ZONE = [0.0, 0.0, 0.45, 0.65] # [x_start, y_start, x_end, y_end] as %

# ── Multi-Zone Monitoring ──
# Each zone gets its own bounding box (as % of frame) and label.
# The detector will process human tracks in each zone independently.
# Override these with your own coordinates for your camera angle.
ZONES_MAP = {
    "desk": {
        "name": "Desk",
        "bbox": [0.0, 0.0, 0.50, 1.0],   # Left half of frame
    },
    "lounge": {
        "name": "Lounge",
        "bbox": [0.50, 0.0, 1.0, 1.0],    # Right half of frame
    },
}

# Night detection
NIGHT_START_HOUR = 19 # 7 PM
NIGHT_END_HOUR = 6    # 6 AM

# ── Human Detection Tuning ─────────────────────────────────────────
# These thresholds only affect the final "person_count" decision.
# If the UI shows a long time in "EMPTY" even when humans are present,
# these values may need to be relaxed for your camera angle/scene.
PERSON_ASPECT_MIN = 0.55               # accept if box_h / box_w >= this
PERSON_MIN_BOX_AREA_RATIO = 0.005     # accept if box_area >= ratio * frame_area
PERSON_ACCEPT_CONF_MIN = 0.45         # accept tracker person without verifier if conf >= this
VERIFIER_IOU_MIN = 0.25               # IoU threshold to merge tracker + verifier boxes

# --- EMAIL ALERT SYSTEM (Safe & Secure) ---
SENDER_EMAIL = os.getenv("VISIONCORE_SENDER_EMAIL", "")
SENDER_PASSWORD = os.getenv("VISIONCORE_SENDER_PASSWORD", "")
RECEIVER_EMAIL = os.getenv("VISIONCORE_RECEIVER_EMAIL", "")

# --- TELEGRAM ALERTS ---
TELEGRAM_ENABLED = os.getenv("VISIONCORE_TELEGRAM_ENABLED", "1") == "1"
TELEGRAM_BOT_TOKEN = os.getenv("VISIONCORE_TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("VISIONCORE_TELEGRAM_CHAT_ID", "")

# --- SMART AUTOMATION ---
# Auto-off: after this many minutes of no activity, mark room as "auto-powered-down"
# and send a notification. Set 0 to disable.
AUTO_OFF_DELAY_MINUTES = 10
# Periodic log frequency (in minutes). Sends a status summary with screenshot
# to email + Telegram. Set 0 to disable. Options: 0, 15, 30, 60, 180.
LOG_FREQUENCY_MINUTES = 0

# --- DATABASE ---
# Prefer Supabase Postgres URL. Example:
# postgresql+psycopg://user:pass@host:5432/postgres
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "")
