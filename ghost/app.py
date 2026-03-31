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
import db

app = Flask(__name__)
monitor = RoomMonitor()
db.init_db()
db.import_csv_to_db_once()

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
_focus_lock = threading.Lock()
_focus_start_ts = None
_focus_seconds = 0

# ── Optional system telemetry ──
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


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
    global _focus_start_ts, _focus_seconds
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
            with _focus_lock:
                if _focus_start_ts is None:
                    _focus_start_ts = time.time()
                _focus_seconds = int(time.time() - _focus_start_ts)
        else:
            with _focus_lock:
                _focus_start_ts = None
                _focus_seconds = 0

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


def _apply_thermal(frame):
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


_trail_prev_gray = None
_trail_canvas = None


def _apply_ghost_trails(frame):
    global _trail_prev_gray, _trail_canvas
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if _trail_prev_gray is None or _trail_canvas is None:
        _trail_prev_gray = gray
        _trail_canvas = np.zeros_like(frame)
        return frame
    diff = cv2.absdiff(gray, _trail_prev_gray)
    _, motion = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    motion_col = cv2.cvtColor(motion, cv2.COLOR_GRAY2BGR)
    motion_col[:, :, 2] = 0
    _trail_canvas = cv2.addWeighted(_trail_canvas, 0.9, motion_col, 0.5, 0)
    out = cv2.addWeighted(frame, 0.85, _trail_canvas, 0.5, 0)
    _trail_prev_gray = gray
    return out


def generate_frames_mode(mode="normal"):
    while True:
        with _camera_lock:
            frame = _latest_frame.copy() if _latest_frame is not None else None
        if frame is None:
            time.sleep(_FRAME_INTERVAL)
            continue
        display = monitor.draw_overlay(frame)
        if mode == "thermal":
            display = _apply_thermal(display)
        elif mode == "trails":
            display = _apply_ghost_trails(display)
        ret, buffer = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ret:
            time.sleep(_FRAME_INTERVAL)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
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

@app.route('/bill')
def bill_page():
    return render_template('bill.html')

@app.route('/video_feed')
def video_feed():
    filter_mode = request.args.get('filter', 'normal')
    if filter_mode == 'thermal':
        return Response(generate_frames_mode("thermal"), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif filter_mode == 'trails':
        return Response(generate_frames_mode("trails"), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_thermal')
def video_feed_thermal():
    return Response(generate_frames_mode("thermal"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_trails')
def video_feed_trails():
    return Response(generate_frames_mode("trails"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/evidence')
def api_evidence():
    evidence_list = db.list_evidence() if db.is_ready() else []
    items = []
    for e in evidence_list:
        if e.get('snapshot_path'):
            filename = os.path.basename(e['snapshot_path'])
            items.append({
                "timestamp": e['timestamp'],
                "duration_seconds": e['duration_seconds'],
                "money_wasted": e['money_wasted'],
                "snapshot_url": f"/evidence/{filename}",
                "room": e['room']
            })
    return jsonify({"items": items})

@app.route('/evidence/<filename>')
def serve_evidence(filename):
    evidence_dir = os.path.join(os.path.dirname(__file__), '..', 'evidence')
    return send_file(os.path.join(evidence_dir, filename))

@app.route('/api/projection')
def energy_projection():
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        return jsonify({"projected_hours": 0, "projected_cost": 0, "error": "sklearn not installed"}), 500
        
    import numpy as np
    from collections import defaultdict
    daily_waste = defaultdict(float)
    entries = db.list_history(limit=5000) if db.is_ready() else []
    for row in entries:
        ts = row.get('Timestamp', '')
        if ts:
            day = ts[:10]
            daily_waste[day] += float(row.get('Duration_Seconds', 0) or 0)
            
    X = np.arange(7).reshape(-1, 1)
    y = []
    for i in range(6, -1, -1):
        d = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        y.append(daily_waste.get(d, 0) / 3600.0)
        
    model = LinearRegression()
    model.fit(X, y)
    next_pred = max(0, model.predict([[7]])[0])
    cost_pred = next_pred * (config.BULB_WATTAGE / 1000.0) * config.ELECTRICITY_RATE
    return jsonify({"projected_hours": round(next_pred, 2), "projected_cost": round(cost_pred, 2)})

@app.route('/api/daily_bill')
def daily_bill():
    today = datetime.now().strftime('%Y-%m-%d')
    total_waste = 0.0
    total_cost = 0.0
    entries = db.list_history(limit=5000) if db.is_ready() else []
    items = []
    for row in entries:
        ts = row.get('Timestamp', '')
        if ts.startswith(today):
            dur = float(row.get('Duration_Seconds', 0) or 0)
            cost = float(row.get('Money_Wasted', 0) or 0)
            total_waste += dur
            total_cost += cost
            items.append({
                "time": ts[11:],
                "duration": round(dur, 2),
                "cost": round(cost, 2),
                "zone": row.get('Zone', 'default')
            })
    return jsonify({
        "date": today,
        "total_hours": round(total_waste/3600, 2),
        "total_cost": round(total_cost, 2),
        "items": items
    })

@app.route('/api/leaderboard')
def leaderboard_api():
    today = datetime.now().strftime('%Y-%m-%d')
    user_waste_sec = 0.0
    entries = db.list_history(limit=5000) if db.is_ready() else []
    for row in entries:
        ts = row.get('Timestamp', '')
        if ts.startswith(today):
            user_waste_sec += float(row.get('Duration_Seconds', 0) or 0)
            
    user_score = max(0, 100 - int(user_waste_sec/60))
    leaderboard = [
        {"name": "Alpha Tech Office", "score": 95},
        {"name": "Desk 4", "score": 88},
        {"name": "Lounge Area", "score": 75},
        {"name": "Your Desk (You)", "score": user_score, "is_user": True}
    ]
    leaderboard.sort(key=lambda x: x["score"], reverse=True)
    return jsonify({"leaderboard": leaderboard})


# ── API Routes ──

@app.route('/status')
def status():
    with _focus_lock:
        focus_seconds = int(_focus_seconds)
    milestones = {
        "1m": focus_seconds >= 60,
        "5m": focus_seconds >= 300,
        "30m": focus_seconds >= 1800,
    }
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
        "verifier_active": getattr(getattr(monitor, "_verifier_thread", None), "is_alive", lambda: False)(),
        "focus_seconds": focus_seconds,
        "focus_milestones": milestones,
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
    entries = db.list_history(limit=50) if db.is_ready() else []
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

    # Fetch from DB.
    fieldnames = ['Timestamp', 'Room', 'Duration_Seconds', 'Status', 'Money_Wasted']
    rows = []
    entries = db.list_history(limit=100000) if db.is_ready() else []
    for row in entries:
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
        rows.append(row)

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
    target_date_str = request.args.get('date', '').strip()
    target_date = None
    if target_date_str:
        try:
            target_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
        except Exception:
            target_date = None

    if db.is_ready():
        return jsonify(db.heatmap_cells(target_date=target_date))
    return jsonify({"cells": [], "max_seconds": 0})

@app.route('/api/monthly_summary')
def monthly_summary():
    """Return aggregated energy-waste cost summary for the current calendar month."""
    now = datetime.now()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    days_elapsed = max(1, (now.date() - month_start.date()).days + 1)

    total_alerts = 0
    total_waste_seconds = 0.0
    total_money_wasted_inr = 0.0

    entries = db.list_history(limit=200000) if db.is_ready() else []
    for row in entries:
        if (row.get('Status') or '').strip() != 'ALERT_SENT':
            continue
        ts = (row.get('Timestamp') or '').strip()
        try:
            dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
        except Exception:
            continue
        if dt < month_start or dt.date() > now.date() or dt.month != now.month or dt.year != now.year:
            continue
        dur = float(row.get('Duration_Seconds', 0) or 0)
        total_alerts += 1
        total_waste_seconds += dur
        total_money_wasted_inr += float(row.get('Money_Wasted', 0) or 0)

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
    from collections import defaultdict

    daily_waste = defaultdict(float)  # date → total waste seconds

    entries = db.list_history(limit=200000) if db.is_ready() else []
    for row in entries:
        ts = row.get('Timestamp', '')
        dur = float(row.get('Duration_Seconds', 0) or 0)
        day = ts[:10] if len(ts) >= 10 else 'Unknown'
        daily_waste[day] += dur

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
    from collections import defaultdict
    
    total_seconds = 0.0
    incident_count = 0
    daily_totals = defaultdict(float)
    
    entries = db.list_history(limit=200000) if db.is_ready() else []
    for row in entries:
        dur = float(row.get('Duration_Seconds', 0) or 0)
        total_seconds += dur
        incident_count += 1
        ts = row.get('Timestamp', '')
        day = ts[:10] if len(ts) >= 10 else 'Unknown'
        daily_totals[day] += dur
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


@app.route('/api/evidence')
def evidence():
    items = db.list_evidence(limit=80) if db.is_ready() else []
    for i in items:
        p = i.get("snapshot_path")
        if p:
            i["snapshot_url"] = f"/evidence_file/{os.path.basename(p)}"
    return jsonify({"items": items})


@app.route('/evidence_file/<filename>')
def evidence_file(filename):
    evidence_dir = os.path.join(os.path.dirname(__file__), 'evidence')
    path = os.path.join(evidence_dir, filename)
    if not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
    return send_file(path)


@app.route('/api/projection')
def projection():
    rows = db.list_history(limit=200000) if db.is_ready() else []
    by_day = {}
    for r in rows:
        d = (r.get("Timestamp") or "")[:10]
        by_day[d] = by_day.get(d, 0.0) + float(r.get("Duration_Seconds", 0) or 0)
    dates = sorted(by_day.keys())[-7:]
    y = [by_day[d] / 3600.0 for d in dates]
    pred = 0.0
    if len(y) >= 2:
        try:
            from sklearn.linear_model import LinearRegression
            import numpy as _np
            X = _np.array(list(range(len(y)))).reshape(-1, 1)
            model = LinearRegression().fit(X, _np.array(y))
            pred = max(0.0, float(model.predict(_np.array([[len(y)]]))[0]))
        except Exception:
            pred = sum(y) / len(y)
    elif len(y) == 1:
        pred = y[0]
    money = pred * (_BULB_WATTAGE / 1000.0) * _ELECTRICITY_RATE
    return jsonify({
        "predicted_waste_hours": round(pred, 2),
        "predicted_money_wasted": round(money, 2),
        "training_days": len(y),
    })


@app.route('/api/leaderboard')
def leaderboard():
    rows = db.list_history(limit=200000) if db.is_ready() else []
    today = datetime.now().strftime('%Y-%m-%d')
    my_waste_seconds = 0.0
    for r in rows:
        ts = r.get("Timestamp", "")
        if ts.startswith(today):
            my_waste_seconds += float(r.get("Duration_Seconds", 0) or 0)
    my_hours = my_waste_seconds / 3600.0
    seed = int(datetime.now().strftime("%Y%m%d"))
    competitors = [
        {"name": "EcoLabs-A", "waste_hours": max(0.0, my_hours * (0.8 + (seed % 7) * 0.03))},
        {"name": "GreenNode-B", "waste_hours": max(0.0, my_hours * (1.0 + (seed % 5) * 0.02))},
        {"name": "SmartGrid-C", "waste_hours": max(0.0, my_hours * (0.9 + (seed % 9) * 0.015))},
    ]
    scores = [{"name": "You", "waste_hours": my_hours}] + competitors
    for s in scores:
        s["eco_score"] = max(0, round(100 - s["waste_hours"] * 20))
    scores.sort(key=lambda x: x["waste_hours"])
    rank = next((i + 1 for i, s in enumerate(scores) if s["name"] == "You"), len(scores))
    return jsonify({"rank": rank, "total": len(scores), "scores": scores})


@app.route('/bill')
def bill():
    target = request.args.get("date", datetime.now().strftime('%Y-%m-%d'))
    rows = db.list_history(limit=200000) if db.is_ready() else []
    entries = [r for r in rows if (r.get("Timestamp") or "").startswith(target)]
    total_seconds = sum(float(r.get("Duration_Seconds", 0) or 0) for r in entries)
    total_money = sum(float(r.get("Money_Wasted", 0) or 0) for r in entries)
    return render_template(
        "bill.html",
        bill_date=target,
        room=config.ROOM_NAME,
        entries=entries,
        total_seconds=round(total_seconds, 2),
        total_money=round(total_money, 2),
        total_kwh=round((total_seconds / 3600.0) * (_BULB_WATTAGE / 1000.0), 3),
    )


@app.route('/api/zones')
def zones():
    """Return configured monitoring zones from ZONES_MAP."""
    zone_map = getattr(config, 'ZONES_MAP', {})
    zone_list = []
    for zid, zdata in zone_map.items():
        zone_list.append({
            "id": zid,
            "name": zdata.get("name", zid),
            "bbox": zdata.get("bbox", [0.0, 0.0, 1.0, 1.0]),
        })
    if not zone_list:
        zone_list = [{"id": "default", "name": "Full Frame", "bbox": [0.0, 0.0, 1.0, 1.0]}]
    return jsonify({"zones": zone_list})


@app.route('/api/focus')
def focus_tracker():
    """Return live focus tracker stats."""
    with _focus_lock:
        secs = _focus_seconds
        
    milestones = {
        "1m": secs >= 60,
        "5m": secs >= 300,
        "30m": secs >= 1800
    }
    
    return jsonify({
        "focus_seconds": secs,
        "milestones": milestones,
        "is_focused": secs > 0
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
