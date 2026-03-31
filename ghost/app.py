from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import numpy as np
import time
import json
import os
import threading
from datetime import datetime, timedelta
import config
from logic.detector import RoomMonitor
from logic.report_generator import generate_daily_report

app = Flask(__name__)
monitor = RoomMonitor()

# ── Reduce OpenCV backend log spam (Windows) ──
try:
    # OpenCV 4.x: suppress INFO/WARN noise from camera backends
    if hasattr(cv2, "setLogLevel") and hasattr(cv2, "LOG_LEVEL_ERROR"):
        cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
    elif hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

# ── Persistent Settings File ──
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), 'settings.json')
_start_time = time.time()
_total_detections = 0

# ── Thread-safe shared state ──
_camera_lock = threading.Lock()
_latest_frame = None
_camera_running = False
_camera_stop_event = threading.Event()
_active_camera_source = None
_TARGET_FPS = 30
_FRAME_INTERVAL = 1.0 / _TARGET_FPS

# ── Energy Savings Constants ──
_BULB_WATTAGE = getattr(config, "BULB_WATTAGE", 200)          # Watts (room lighting baseline)
_ELECTRICITY_RATE = getattr(config, "ELECTRICITY_RATE", 8.0) # ₹ per kWh (Indian average)
_energy_lock = threading.Lock()
_waste_seconds_today = 0.0   # accumulated empty-with-light seconds today
_session_waste_seconds = 0.0 # accumulated since app start
_last_waste_check = time.time()
_savings_date = datetime.now().strftime('%Y-%m-%d')  # reset tracker at midnight

# ── Occupancy History (10 min = 600 data points at 1 sample/sec) ──
from collections import deque
_occupancy_history = deque(maxlen=600)
_occupancy_timestamps = deque(maxlen=600)

# ── Optional system telemetry ──
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

def migrate_csv_schema():
    """
    One-time migration: ensure `reports/energy_audit.csv` includes Money_Wasted.
    Idempotent: if column already exists, does nothing.
    """
    import csv

    log_file = os.path.join(os.path.dirname(__file__), 'reports', 'energy_audit.csv')
    if not os.path.exists(log_file):
        return

    try:
        with open(log_file, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)

        if not header or 'Money_Wasted' in header:
            return

        # Re-read as dicts using the existing header.
        rows = []
        with open(log_file, 'r', newline='') as f:
            dict_reader = csv.DictReader(f)
            for row in dict_reader:
                dur = float(row.get('Duration_Seconds', 0) or 0)
                money_wasted = (dur / 3600.0) * (_BULB_WATTAGE / 1000.0) * _ELECTRICITY_RATE
                row['Money_Wasted'] = round(money_wasted, 2)
                rows.append(row)

        # Rewrite with canonical schema.
        fieldnames = ['Timestamp', 'Room', 'Duration_Seconds', 'Status', 'Money_Wasted']
        with open(log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    'Timestamp': row.get('Timestamp', ''),
                    'Room': row.get('Room', ''),
                    'Duration_Seconds': row.get('Duration_Seconds', 0),
                    'Status': row.get('Status', ''),
                    'Money_Wasted': row.get('Money_Wasted', 0)
                })
    except Exception as e:
        print(f"[MIGRATION] Failed: {e}")


# Migrate CSV schema before any threads start (so UI & exports are consistent).
migrate_csv_schema()

def _looks_like_valid_camera_frame(frame):
    """Reject clearly broken/corrupt frames (e.g. flat orange virtual-camera feed)."""
    if frame is None or frame.size == 0:
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Near-flat image usually indicates a broken source frame.
    if float(np.std(gray)) < 6.0:
        return False
    b_mean = float(np.mean(frame[:, :, 0]))
    g_mean = float(np.mean(frame[:, :, 1]))
    r_mean = float(np.mean(frame[:, :, 2]))
    channel_spread = max(b_mean, g_mean, r_mean) - min(b_mean, g_mean, r_mean)
    # Very strong single-channel cast + very low texture = likely invalid feed.
    if channel_spread > 85.0 and float(np.std(gray)) < 12.0:
        return False
    return True


def _probe_camera_source(src):
    """Open camera source and validate it with a few sample frames."""
    cam = _try_open_camera(src)
    if not cam.isOpened():
        return None
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cam.set(cv2.CAP_PROP_FPS, 30)

    valid_frames = 0
    for _ in range(20):
        ok, frame = cam.read()
        if ok and _looks_like_valid_camera_frame(frame):
            valid_frames += 1
        time.sleep(0.01)
        if valid_frames >= 3:
            return cam
    cam.release()
    return None


def _try_open_camera(src):
    # Windows: try multiple backends because different systems enumerate cameras differently.
    if os.name != "nt":
        return cv2.VideoCapture(src)

    backend_candidates = [None]
    if hasattr(cv2, "CAP_DSHOW"):
        backend_candidates.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):
        backend_candidates.append(cv2.CAP_MSMF)
    if hasattr(cv2, "CAP_ANY"):
        backend_candidates.append(cv2.CAP_ANY)

    last_cam = None
    for backend in backend_candidates:
        cam = cv2.VideoCapture(src) if backend is None else cv2.VideoCapture(src, backend)
        last_cam = cam
        if cam.isOpened():
            return cam
        try:
            cam.release()
        except Exception:
            pass
    return last_cam


def _camera_thread():
    """Capture frames in background — never blocks anything."""
    global _latest_frame, _camera_running, _active_camera_source

    # Try primary configured source first; if it fails, auto-scan a few common webcam indexes.
    # This prevents the UI MJPEG stream from staying black when CAM index is wrong.
    source = getattr(config, "CAMERA_SOURCE", 0)

    camera = _probe_camera_source(source)
    if camera is None:
        # If `CAMERA_SOURCE` is an index, scan nearby indexes for a working webcam.
        candidates = None
        if isinstance(source, int):
            candidates = list(range(0, 6))
        elif isinstance(source, str) and source.isdigit():
            candidates = list(range(0, 6))

        if candidates:
            for idx in candidates:
                test_cam = _probe_camera_source(idx)
                if test_cam is not None:
                    camera = test_cam
                    source = idx
                    break

    if camera is None:
        _camera_running = False
        print(f"[CAMERA] Opened: False. No usable camera source found (source={getattr(config, 'CAMERA_SOURCE', None)}).")
        return

    _camera_running = True
    _active_camera_source = source
    _camera_stop_event.clear()
    print(f"[CAMERA] Opened: True (source={source}), Target: 30fps")

    while _camera_running and not _camera_stop_event.is_set():
        success, frame = camera.read()
        if success:
            with _camera_lock:
                _latest_frame = frame
        else:
            time.sleep(0.005)

    camera.release()
    _camera_running = False


def _restart_camera_thread():
    """Safely restart camera thread when camera source changes."""
    global _cam_thread, _latest_frame
    _camera_stop_event.set()
    if '_cam_thread' in globals() and _cam_thread.is_alive():
        _cam_thread.join(timeout=1.5)

    with _camera_lock:
        _latest_frame = None

    _cam_thread = threading.Thread(target=_camera_thread, daemon=True)
    _cam_thread.start()


def _ai_thread():
    """Run AI detection asynchronously — doesn't block the video stream.
    The AI processes frames as fast as it can. The overlay data is stored
    in monitor._overlay_* and drawn by the stream thread independently."""
    global _total_detections
    while True:
        with _camera_lock:
            frame = _latest_frame.copy() if _latest_frame is not None else None
        if frame is None:
            time.sleep(0.01)
            continue

        # AI processes at maximum possible speed (no drawing here)
        monitor.process_frame(frame)
        if monitor.person_count > 0:
            _total_detections += 1

        # ── Record occupancy sample ──
        now = time.time()
        _occupancy_history.append(monitor.person_count)
        _occupancy_timestamps.append(now)


# ── Energy Waste Tracker Thread ──
def _energy_tracker_thread():
    """Runs every second: if room is empty + lights on, accumulate waste time."""
    global _waste_seconds_today, _session_waste_seconds, _last_waste_check, _savings_date
    while True:
        time.sleep(1)
        now_ts = time.time()
        now_date = datetime.now().strftime('%Y-%m-%d')

        with _energy_lock:
            # Reset daily counter at midnight
            if now_date != _savings_date:
                _waste_seconds_today = 0.0
                _savings_date = now_date

            # If energy is being wasted (empty room + lights on)
            if monitor.is_energy_wasted:
                elapsed = now_ts - _last_waste_check
                _waste_seconds_today += elapsed
                _session_waste_seconds += elapsed

            _last_waste_check = now_ts


# ── Daily Report Scheduler ──
def _daily_report_scheduler():
    """Background thread: auto-generate PDF report at 23:55 each day."""
    while True:
        now = datetime.now()
        # Next trigger at 23:55 today (or tomorrow if already past)
        target = now.replace(hour=23, minute=55, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        wait_seconds = (target - now).total_seconds()
        print(f"[REPORT SCHEDULER] Next daily report in {wait_seconds/3600:.1f}h at {target.strftime('%H:%M')}")
        time.sleep(wait_seconds)

        # Generate report for today
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            filepath = generate_daily_report(target_date=today, room_name=config.ROOM_NAME)
            print(f"[REPORT SCHEDULER] Auto-generated: {filepath}")
        except Exception as e:
            print(f"[REPORT SCHEDULER] Failed: {e}")


# Start threads
_cam_thread = threading.Thread(target=_camera_thread, daemon=True)
_cam_thread.start()
_ai_thread_obj = threading.Thread(target=_ai_thread, daemon=True)
_ai_thread_obj.start()
_energy_tracker = threading.Thread(target=_energy_tracker_thread, daemon=True)
_energy_tracker.start()
_report_thread = threading.Thread(target=_daily_report_scheduler, daemon=True)
_report_thread.start()
time.sleep(0.5)


def load_settings():
    """Load settings from JSON file and apply to config module."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                saved = json.load(f)
            if 'receiver_email' in saved:
                config.RECEIVER_EMAIL = saved['receiver_email']
            if 'room_name' in saved:
                config.ROOM_NAME = saved['room_name']
            if 'alert_delay' in saved:
                config.ALERT_DELAY_SECONDS = int(saved['alert_delay'])
            if 'camera_source' in saved:
                raw_source = saved['camera_source']
                if isinstance(raw_source, str) and raw_source.isdigit():
                    raw_source = int(raw_source)
                config.CAMERA_SOURCE = raw_source
            print(f"[SETTINGS] Loaded from {SETTINGS_FILE}")
            print(f"  → Receiver: {config.RECEIVER_EMAIL}")
            print(f"  → Room: {config.ROOM_NAME}")
            print(f"  → Alert Delay: {config.ALERT_DELAY_SECONDS}s")
            print(f"  → Camera Source: {getattr(config, 'CAMERA_SOURCE', 0)}")
        except Exception as e:
            print(f"[SETTINGS] Failed to load: {e}")


def save_settings():
    """Save current config settings to JSON file."""
    data = {
        'receiver_email': config.RECEIVER_EMAIL,
        'room_name': config.ROOM_NAME,
        'alert_delay': config.ALERT_DELAY_SECONDS,
        'camera_source': getattr(config, 'CAMERA_SOURCE', 0)
    }
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"[SETTINGS] Failed to save: {e}")
        return False


# Load saved settings on startup
load_settings()
# Apply saved camera source immediately after loading settings.
_restart_camera_thread()


def generate_frames():
    """Stream at 30fps.
    
    KEY ARCHITECTURE: We always grab the LATEST raw camera frame,
    overlay the latest AI boxes onto it, and stream it.
    The AI thread updates overlay data independently.
    This means the video is always smooth even if AI is slow.
    """
    placeholder = None
    while True:
        with _camera_lock:
            frame = _latest_frame.copy() if _latest_frame is not None else None

        if frame is None:
            # Keep MJPEG stream responsive even when camera isn't available.
            # Otherwise the browser can render a permanently black area.
            if placeholder is None:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                # Background panel for readability (camera missing scenario).
                cv2.rectangle(placeholder, (10, 200), (630, 300), (0, 0, 0), -1)
                cv2.rectangle(placeholder, (10, 200), (630, 300), (148, 163, 184), 2)  # border
                cv2.putText(
                    placeholder,
                    "Camera not available",
                    (30, 255),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (11, 158, 245),  # orange-ish (BGR) to match HUD palette
                    2,
                    cv2.LINE_AA,
                )
            display = placeholder
        else:
            # Draw latest AI detections onto the raw frame
            display = monitor.draw_overlay(frame)

        ret, buffer = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 75])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(_FRAME_INTERVAL)


# ── Page Routes ──

@app.route('/')
def home():
    """Project landing page."""
    return render_template('home.html')

@app.route('/monitor')
def monitor_page():
    return render_template('monitor.html')


@app.route('/history')
def history_page():
    return render_template('history.html')


@app.route('/reports')
def reports_page():
    return render_template('reports.html')


@app.route('/settings')
def settings_page():
    return render_template('settings.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ── API Routes ──

@app.route('/status')
def status():
    return jsonify({
        "person_count": monitor.person_count,
        "light_status": monitor.light_status,
        # For now light_type mirrors light_status but is kept for backward compatibility.
        "light_type": monitor.light_status,
        "luminance": getattr(monitor, "luminance", None),
        "light_debug": getattr(monitor, "light_debug", {}),
        "is_energy_wasted": monitor.is_energy_wasted,
        "time_since_presence": int(time.time() - monitor.last_seen_time),
        "ai_fps": monitor._ai_fps,
        "detection_confidence": getattr(monitor, "detection_confidence", None),
        "verifier_active": getattr(getattr(monitor, "_verifier_thread", None), "is_alive", lambda: False)()
    })

@app.route('/health')
def health():
    """Lightweight health endpoint for UI telemetry."""
    camera_running = bool(_camera_running) and _cam_thread.is_alive()
    camera_fps = float(_TARGET_FPS) if camera_running else 0.0

    ai_loaded = getattr(monitor, "model_primary", None) is not None
    ai_fps = float(getattr(monitor, "_ai_fps", 0.0) or 0.0)

    verifier_queue_depth = 1 if getattr(monitor, "_verifier_frame", None) is not None else 0
    energy_running = _energy_tracker.is_alive() if '_energy_tracker' in globals() else False

    if not camera_running:
        status_str = "error"
    elif ai_loaded and ai_fps >= 1.0:
        status_str = "ok"
    else:
        status_str = "degraded"

    verifier_active = getattr(getattr(monitor, "_verifier_thread", None), "is_alive", lambda: False)()
    engine = "YOLOv8n + YOLOv8m (CUDA)" if getattr(monitor, "_device", "cpu") == "cuda" else "YOLOv8n + YOLOv8m (CPU)"

    return jsonify({
        "status": status_str,
        "camera": {
            "running": camera_running,
            "fps": round(camera_fps, 1)
        },
        "ai_model": {
            "loaded": bool(ai_loaded),
            "engine": engine,
            "fps": round(ai_fps, 1)
        },
        "verifier": {
            "queue_depth": int(verifier_queue_depth)
        },
        "energy_tracker": {
            "running": bool(energy_running),
            "waste_seconds_today": int(_waste_seconds_today)
        },
        "uptime_seconds": int(time.time() - _start_time)
    })

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Return current settings."""
    return jsonify({
        "receiver_email": config.RECEIVER_EMAIL,
        "room_name": config.ROOM_NAME,
        "alert_delay": config.ALERT_DELAY_SECONDS,
        "camera_source": getattr(config, "CAMERA_SOURCE", 0)
    })

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update settings and persist to disk."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    if 'receiver_email' in data:
        email = data['receiver_email'].strip()
        if '@' not in email or '.' not in email:
            return jsonify({"error": "Invalid email address"}), 400
        config.RECEIVER_EMAIL = email

    if 'room_name' in data:
        config.ROOM_NAME = data['room_name'].strip()

    if 'alert_delay' in data:
        try:
            config.ALERT_DELAY_SECONDS = int(data['alert_delay'])
        except ValueError:
            pass

    camera_restarted = False
    if 'camera_source' in data:
        raw_source = data['camera_source']
        if isinstance(raw_source, str):
            raw_source = raw_source.strip()
            new_source = int(raw_source) if raw_source.isdigit() else raw_source
        else:
            new_source = raw_source

        if new_source != getattr(config, "CAMERA_SOURCE", 0):
            config.CAMERA_SOURCE = new_source
            _restart_camera_thread()
            camera_restarted = True

    monitor.alert_sent = False
    print(
        f"[SETTINGS] Updated → Email: {config.RECEIVER_EMAIL}, Room: {config.ROOM_NAME}, "
        f"Delay: {config.ALERT_DELAY_SECONDS}s, Camera: {getattr(config, 'CAMERA_SOURCE', 0)}"
    )

    saved = save_settings()

    return jsonify({
        "success": True,
        "saved_to_disk": saved,
        "receiver_email": config.RECEIVER_EMAIL,
        "room_name": config.ROOM_NAME,
        "alert_delay": config.ALERT_DELAY_SECONDS,
        "camera_source": getattr(config, "CAMERA_SOURCE", 0),
        "camera_restarted": camera_restarted
    })


@app.route('/api/test_email', methods=['POST'])
def test_email():
    """Send a test email to verify the receiver address works."""
    success, message = monitor.send_test_email()
    return jsonify({
        "success": success,
        "message": message,
        "receiver_email": config.RECEIVER_EMAIL
    })


@app.route('/api/history')
def history():
    """Return energy audit log entries."""
    log_file = os.path.join(os.path.dirname(__file__), 'reports', 'energy_audit.csv')
    entries = []
    if os.path.exists(log_file):
        try:
            import csv
            with open(log_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    entries.append(row)
            entries = entries[-50:]
        except Exception:
            pass
    return jsonify({"entries": entries, "total": len(entries)})

@app.route('/api/export_csv', methods=['GET'])
def export_csv():
    """
    Export `energy_audit.csv` as a downloadable CSV, optionally filtered by date range.
    Query params: ?from=YYYY-MM-DD&to=YYYY-MM-DD
    """
    import csv
    from io import StringIO

    log_file = os.path.join(os.path.dirname(__file__), 'reports', 'energy_audit.csv')

    from_str = request.args.get('from', '').strip()
    to_str = request.args.get('to', '').strip()

    from_date = None
    to_date = None
    try:
        if from_str:
            from_date = datetime.strptime(from_str, '%Y-%m-%d').date()
        if to_str:
            to_date = datetime.strptime(to_str, '%Y-%m-%d').date()
    except Exception:
        from_date = None
        to_date = None

    # Read CSV (if missing, return header only).
    fieldnames = ['Timestamp', 'Room', 'Duration_Seconds', 'Status', 'Money_Wasted']
    rows = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    fieldnames = list(reader.fieldnames)
                    if 'Money_Wasted' not in fieldnames:
                        fieldnames.append('Money_Wasted')

                for row in reader:
                    ts = (row.get('Timestamp') or '').strip()
                    try:
                        dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                    except Exception:
                        continue

                    row_date = dt.date()
                    if from_date and row_date < from_date:
                        continue
                    if to_date and row_date > to_date:
                        continue

                    # Backfill Money_Wasted if it is missing/unparseable.
                    money_val = row.get('Money_Wasted', None)
                    try:
                        money_float = float(money_val)
                        row['Money_Wasted'] = round(money_float, 2)
                    except Exception:
                        dur = float(row.get('Duration_Seconds', 0) or 0)
                        money_wasted = (dur / 3600.0) * (_BULB_WATTAGE / 1000.0) * _ELECTRICITY_RATE
                        row['Money_Wasted'] = round(money_wasted, 2)

                    rows.append(row)
        except Exception:
            rows = []

    # Produce output CSV
    out = StringIO()
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()

    for row in rows:
        writer.writerow({fn: row.get(fn, '') for fn in fieldnames})

    resp = Response(out.getvalue(), mimetype='text/csv')
    resp.headers['Content-Disposition'] = 'attachment; filename=visioncore_audit_export.csv'
    return resp

@app.route('/api/heatmap_data')
def heatmap_data():
    """Return hour×weekday waste totals from the audit CSV.

    Optional query:
        ?date=YYYY-MM-DD  -> restrict aggregation to a single calendar date.
    """
    import csv
    from collections import defaultdict

    log_file = os.path.join(os.path.dirname(__file__), 'reports', 'energy_audit.csv')

    cell_totals = defaultdict(float)  # (weekday, hour) -> seconds
    max_seconds = 0.0

    target_date_str = request.args.get('date', '').strip()
    target_date = None
    if target_date_str:
        try:
            target_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
        except Exception:
            target_date = None

    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if (row.get('Status') or '').strip() != 'ALERT_SENT':
                        continue

                    ts = (row.get('Timestamp') or '').strip()
                    try:
                        dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                    except Exception:
                        continue

                    if target_date is not None and dt.date() != target_date:
                        continue

                    dur = float(row.get('Duration_Seconds', 0) or 0)
                    key = (dt.weekday(), dt.hour)  # weekday: 0=Mon..6=Sun
                    cell_totals[key] += dur
        except Exception:
            pass

    # Compute max after accumulation
    if cell_totals:
        max_seconds = max(cell_totals.values())

    cells = []
    for (day, hour), seconds in cell_totals.items():
        if seconds > 0:
            cells.append({"day": int(day), "hour": int(hour), "seconds": round(seconds, 2)})

    return jsonify({"cells": cells, "max_seconds": round(max_seconds, 2)})

@app.route('/api/monthly_summary')
def monthly_summary():
    """Return aggregated energy-waste cost summary for the current calendar month."""
    import csv

    log_file = os.path.join(os.path.dirname(__file__), 'reports', 'energy_audit.csv')

    now = datetime.now()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    days_elapsed = max(1, (now.date() - month_start.date()).days + 1)

    total_alerts = 0
    total_waste_seconds = 0.0
    total_money_wasted_inr = 0.0

    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if (row.get('Status') or '').strip() != 'ALERT_SENT':
                        continue

                    ts = (row.get('Timestamp') or '').strip()
                    try:
                        dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                    except Exception:
                        continue

                    # Filter to current month only
                    if dt < month_start or dt.date() > now.date() or dt.month != now.month or dt.year != now.year:
                        # The dt.month/year check is redundant with date comparisons, but kept for clarity.
                        continue

                    dur = float(row.get('Duration_Seconds', 0) or 0)
                    total_alerts += 1
                    total_waste_seconds += dur

                    money_val = row.get('Money_Wasted', None)
                    try:
                        money_float = float(money_val)
                        total_money_wasted_inr += money_float
                    except Exception:
                        money_wasted = (dur / 3600.0) * (_BULB_WATTAGE / 1000.0) * _ELECTRICITY_RATE
                        total_money_wasted_inr += money_wasted
        except Exception:
            pass

    total_waste_hours = total_waste_seconds / 3600.0
    avg_daily_waste_minutes = (total_waste_seconds / max(1, float(days_elapsed))) / 60.0

    return jsonify({
        "total_alerts": int(total_alerts),
        "total_waste_hours": round(total_waste_hours, 2),
        "total_money_wasted_inr": round(total_money_wasted_inr, 2),
        "avg_daily_waste_minutes": round(avg_daily_waste_minutes, 2),
        "month_label": now.strftime('%B %Y')
    })


@app.route('/stats')
def stats():
    """Return system runtime stats with hardware info and system telemetry."""
    uptime = int(time.time() - _start_time)
    hours, remainder = divmod(uptime, 3600)
    minutes, seconds = divmod(remainder, 60)

    result = {
        "uptime": f"{hours}h {minutes}m {seconds}s",
        "uptime_seconds": uptime,
        "total_frames_with_humans": _total_detections,
        "current_person_count": monitor.person_count,
        "alerts_sent": monitor.alert_sent,
        "ai_fps": monitor._ai_fps,
        "hardware": monitor.hw_info,
        "engine": "YOLOv8n (Tracker) + YOLOv8m (Verifier)"
    }

    # ── System telemetry (if psutil available) ──
    if _HAS_PSUTIL:
        try:
            result["system"] = {
                "cpu_percent": psutil.cpu_percent(interval=0),
                "ram_percent": psutil.virtual_memory().percent,
                "ram_used_gb": round(psutil.virtual_memory().used / 1e9, 1),
                "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
            }
        except Exception:
            pass

    return jsonify(result)

@app.route('/api/energy_savings')
def energy_savings():
    """Return historical energy savings data from audit CSV."""
    import csv
    from collections import defaultdict

    log_file = os.path.join(os.path.dirname(__file__), 'reports', 'energy_audit.csv')
    daily_waste = defaultdict(float)  # date → total waste seconds

    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts = row.get('Timestamp', '')
                    dur = float(row.get('Duration_Seconds', 0))
                    day = ts[:10] if len(ts) >= 10 else 'Unknown'
                    daily_waste[day] += dur
        except Exception:
            pass

    # Last 7 days
    labels = []
    data = []
    date_keys = []
    for i in range(6, -1, -1):
        dt = datetime.now() - timedelta(days=i)
        d = dt.strftime('%Y-%m-%d')
        labels.append(dt.strftime('%a'))
        date_keys.append(d)
        waste_hrs = round(daily_waste.get(d, 0) / 3600, 2)
        data.append(waste_hrs)

    return jsonify({"labels": labels, "data": data, "dates": date_keys})


@app.route('/api/history_stats')
def history_stats():
    """Return aggregated statistics for the History Dashboard."""
    import csv
    from collections import defaultdict
    log_file = os.path.join(os.path.dirname(__file__), 'reports', 'energy_audit.csv')
    
    total_seconds = 0.0
    incident_count = 0
    daily_totals = defaultdict(float)
    
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dur = float(row.get('Duration_Seconds', 0))
                    total_seconds += dur
                    incident_count += 1
                    ts = row.get('Timestamp', '')
                    day = ts[:10] if len(ts) >= 10 else 'Unknown'
                    daily_totals[day] += dur
        except Exception:
            pass
            
    total_hours = total_seconds / 3600
    kwh_wasted = total_hours * (_BULB_WATTAGE / 1000)
    money_wasted = kwh_wasted * _ELECTRICITY_RATE
    
    # Estimate "Money Saved" based on system uptime vs waste
    uptime_hours = (time.time() - _start_time) / 3600
    potential_cost = uptime_hours * (_BULB_WATTAGE / 1000) * _ELECTRICITY_RATE
    money_saved = max(0, potential_cost - money_wasted)

    return jsonify({
        "total_incidents": incident_count,
        "total_waste_hours": round(total_hours, 2),
        "total_money_wasted": round(money_wasted, 2),
        "total_money_saved": round(money_saved, 2),
        "avg_incident_duration": round(total_seconds / incident_count, 1) if incident_count > 0 else 0
    })


@app.route('/api/energy_live')
def energy_live():
    """Real-time energy savings estimator — ₹ and kWh saved today."""
    with _energy_lock:
        waste_today = _waste_seconds_today
        waste_session = _session_waste_seconds

    waste_hours_today = waste_today / 3600
    waste_hours_session = waste_session / 3600

    # Energy = Power × Time; Cost = Energy × Rate
    kwh_today = waste_hours_today * (_BULB_WATTAGE / 1000)
    money_today = kwh_today * _ELECTRICITY_RATE
    kwh_session = waste_hours_session * (_BULB_WATTAGE / 1000)
    money_session = kwh_session * _ELECTRICITY_RATE

    # Current waste rate (₹/hour if lights are on empty now)
    rate_per_hour = (_BULB_WATTAGE / 1000) * _ELECTRICITY_RATE

    return jsonify({
        "today": {
            "waste_seconds": round(waste_today, 1),
            "waste_minutes": round(waste_today / 60, 1),
            "kwh_wasted": round(kwh_today, 3),
            "money_wasted": round(money_today, 2),
        },
        "session": {
            "waste_seconds": round(waste_session, 1),
            "kwh_wasted": round(kwh_session, 3),
            "money_wasted": round(money_session, 2),
        },
        "config": {
            "bulb_wattage": _BULB_WATTAGE,
            "electricity_rate": _ELECTRICITY_RATE,
            "rate_per_hour": round(rate_per_hour, 2),
        },
        "is_wasting_now": monitor.is_energy_wasted
    })


@app.route('/api/occupancy_history')
def occupancy_history():
    """Return last 10 minutes of person_count samples for live chart."""
    # Downsample to ~1 point per 5 seconds for smooth chart (max 120 points)
    counts = list(_occupancy_history)
    timestamps = list(_occupancy_timestamps)

    if len(counts) == 0:
        return jsonify({"labels": [], "data": []})

    # Downsample: take every 5th sample
    step = max(1, len(counts) // 120)
    sampled_counts = counts[::step]
    sampled_times = timestamps[::step]

    now = time.time()
    labels = []
    for t in sampled_times:
        ago = int(now - t)
        if ago < 60:
            labels.append(f"{ago}s ago")
        else:
            labels.append(f"{ago // 60}m {ago % 60}s")

    return jsonify({
        "labels": labels,
        "data": sampled_counts,
        "current": monitor.person_count
    })


@app.route('/api/generate_report', methods=['POST'])
def generate_report():
    """Generate a PDF report on-demand and return download info."""
    data = request.get_json() or {}
    target_date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
    room = data.get('room_name', config.ROOM_NAME)

    try:
        filepath = generate_daily_report(target_date=target_date, room_name=room)
        filename = os.path.basename(filepath)
        return jsonify({
            "success": True,
            "message": f"Report generated for {target_date}",
            "filename": filename,
            "download_url": f"/download_report/{filename}"
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/download_report/<filename>')
def download_report(filename):
    """Download a generated PDF report."""
    reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
    filepath = os.path.join(reports_dir, filename)
    if not os.path.exists(filepath) or not filename.endswith('.pdf'):
        return jsonify({"error": "Report not found"}), 404
    return send_file(filepath, as_attachment=True, download_name=filename)


@app.route('/api/reports')
def list_reports():
    """List all available PDF reports."""
    reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
    reports = []
    if os.path.isdir(reports_dir):
        for f in sorted(os.listdir(reports_dir), reverse=True):
            if f.endswith('.pdf'):
                fpath = os.path.join(reports_dir, f)
                reports.append({
                    "filename": f,
                    "download_url": f"/download_report/{f}",
                    "size_kb": round(os.path.getsize(fpath) / 1024, 1),
                    "created": datetime.fromtimestamp(
                        os.path.getmtime(fpath)).strftime('%Y-%m-%d %H:%M')
                })
    return jsonify({"reports": reports})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
