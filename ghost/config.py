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

# Night detection
NIGHT_START_HOUR = 19 # 7 PM
NIGHT_END_HOUR = 6    # 6 AM

# --- EMAIL ALERT SYSTEM (Safe & Secure) ---
SENDER_EMAIL = "vaish4894@gmail.com"
SENDER_PASSWORD = "ezpi yboc kqbt hmug" 
RECEIVER_EMAIL = "vmaingade21@gmail.com"
