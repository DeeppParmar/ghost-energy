import cv2
import numpy as np
import time
import threading
from datetime import datetime
from ultralytics import YOLO
import sys
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import torch
from collections import deque
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import config
except ImportError:
    from .. import config
try:
    import db
except ImportError:
    from .. import db


def _compute_iou(boxA, boxB):
    """Compute Intersection over Union between two boxes [x1,y1,x2,y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)


class RoomMonitor:
    def __init__(self):
        print("[INIT] ═══════════════════════════════════════")
        print("[INIT] Loading Optimized Detection Pipeline...")

        # ── Auto-detect GPU/CPU ──
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._use_half = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
        vram = f"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB" if torch.cuda.is_available() else 'N/A'

        print(f"[INIT] Device: {self._device.upper()}")
        if torch.cuda.is_available():
            print(f"[INIT] GPU: {gpu_name} ({vram} VRAM)")
            print(f"[INIT] FP16 Half-Precision: ENABLED")
        else:
            print(f"[INIT] No CUDA GPU — using CPU")
            print(f"[INIT] TIP: Install CUDA toolkit + torch-cuda for 5-10x speedup")

        # ── Primary: YOLOv8n (nano) + tracking — ultra fast ──
        self.model_primary = YOLO('yolov8n.pt')
        # ── Secondary: YOLOv8m — runs in background for verification ──
        self.model_secondary = YOLO('yolov8m.pt')

        if self._device == 'cuda':
            self.model_primary.to(self._device)
            self.model_secondary.to(self._device)
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self.model_primary(dummy, verbose=False)
            self.model_secondary(dummy, verbose=False)
            print("[INIT] GPU Warmup Complete ✓")

        self.model_names = self.model_primary.names
        print("[INIT] Pipeline: YOLOv8n (Tracker) + YOLOv8m (Verifier)")
        print("[INIT] ═══════════════════════════════════════")

        # ── State ──
        self.person_count = 0
        self.light_status = "Dark"
        # Numeric luminance value (0-255 avg grayscale) for UI telemetry.
        self.luminance = None
        # Debug telemetry for luminance/light classification.
        self.light_debug = {}
        self.is_energy_wasted = False
        self.alert_sent = False
        # Average confidence (0.0-1.0) from the primary tracker for current detections.
        # Used by the /health and /status endpoints.
        self.detection_confidence = None

        # ── Auto-Off State ──
        # Tracks whether the room has been auto-powered-down due to prolonged inactivity.
        self.auto_off_active = False
        self._auto_off_notified = False
        self._global_last_seen = time.time()

        # ── Multi-Zone State ──
        self.zones_state = {}
        zm = getattr(config, 'ZONES_MAP', {})
        if zm:
            for zid, zdata in zm.items():
                self.zones_state[zid] = {
                    "name": zdata.get("name", zid),
                    "bbox": zdata.get("bbox", [0.0, 0.0, 1.0, 1.0]),
                    "person_count": 0,
                    "last_seen_time": time.time(),
                    "is_energy_wasted": False,
                    "alert_sent": False
                }
        else:
            self.zones_state["default"] = {
                "name": "Default",
                "bbox": getattr(config, 'WINDOW_ZONE', [0.0, 0.0, 1.0, 1.0]),
                "person_count": 0,
                "last_seen_time": time.time(),
                "is_energy_wasted": False,
                "alert_sent": False
            }

        # ── Inference settings ──
        self._infer_size = (320, 320) if self._device == 'cuda' else (256, 256)

        # ── Performance tracking ──
        self._ai_fps = 0.0
        self._fps_ring = deque(maxlen=30)

        # ── Temporal smoothing for stable count ──
        self._count_history = deque(maxlen=10)
        self._previous_person_count = 0

        # ── Overlay data (shared with stream thread) ──
        self._overlay_lock = threading.Lock()
        self._overlay_humans = []       # [(x1,y1,x2,y2,conf,track_id), ...]
        self._overlay_objects = []      # [{'coords':..., 'name':..., 'conf':...}, ...]
        self._overlay_scale = (1.0, 1.0)
        # Camera HUD telemetry snapshot for overlay drawing.
        self._overlay_luminance = None  # raw luminance (0-255) from grayscale mean
        self._overlay_light_status = "Dark"

        # ── Async verifier state ──
        self._verifier_lock = threading.Lock()
        self._verifier_confirmed = []   # verified human boxes from YOLOv8m
        self._verifier_frame = None
        self._verifier_busy = False
        self._verifier_thread = threading.Thread(target=self._verifier_loop, daemon=True)
        self._verifier_thread.start()

        # ── Hardware info ──
        self.hw_info = {
            'device': self._device,
            'gpu_name': gpu_name,
            'vram': vram,
            'fp16': self._use_half,
            'infer_size': self._infer_size[0]
        }

    # ── Background Verifier Loop ──────────────────────────────────────

    def _verifier_loop(self):
        """YOLOv8m runs continuously in background, verifying detections."""
        while True:
            frame = None
            with self._verifier_lock:
                if self._verifier_frame is not None:
                    frame = self._verifier_frame.copy()
                    self._verifier_frame = None
                    self._verifier_busy = True

            if frame is None:
                time.sleep(0.05)
                continue

            try:
                results = self.model_secondary(
                    frame, conf=0.25, iou=0.5, verbose=False,
                    half=self._use_half, device=self._device,
                    classes=[0]  # only detect persons
                )
                humans = []
                for r in results:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        coords = box.xyxy[0].cpu().numpy().tolist()
                        humans.append(coords + [conf])

                with self._verifier_lock:
                    self._verifier_confirmed = humans
                    self._verifier_busy = False
            except Exception as e:
                print(f"[VERIFIER] Error: {e}")
                with self._verifier_lock:
                    self._verifier_busy = False

    # ── Light Analysis ────────────────────────────────────────────────

    def _is_night_time(self):
        hour = datetime.now().hour
        return hour >= config.NIGHT_START_HOUR or hour < config.NIGHT_END_HOUR

    def analyze_light(self, frame):
        """Advanced Light Classification: LED vs Sunlight."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        # Persist raw luminance so the UI can show exactly what the AI "sees".
        self.luminance = brightness
        is_night = self._is_night_time()
        self.light_debug = {
            "brightness_mean": brightness,
            "brightness_threshold": float(config.BRIGHTNESS_THRESHOLD),
            "is_night_time": is_night,
            "window_brightness": None,
            "blue_red_ratio": None,
        }
        if brightness < config.BRIGHTNESS_THRESHOLD:
            self.light_status = "Dark"
            self.light_debug["decision"] = "Dark (below threshold)"
            return self.light_status

        if is_night:
            self.light_status = "Artificial Light"
            self.light_debug["decision"] = "Artificial (night-time)"
            return self.light_status

        h, w = frame.shape[:2]
        x1, y1 = int(config.WINDOW_ZONE[0] * w), int(config.WINDOW_ZONE[1] * h)
        x2, y2 = int(config.WINDOW_ZONE[2] * w), int(config.WINDOW_ZONE[3] * h)
        window_region = frame[y1:y2, x1:x2]

        if window_region.size > 0:
            b_mean = float(np.mean(window_region[:, :, 0]))
            r_mean = float(np.mean(window_region[:, :, 2]))
            ratio = b_mean / (r_mean + 1e-6)
            window_brightness = float(np.mean(cv2.cvtColor(window_region, cv2.COLOR_BGR2GRAY)))
            self.light_debug["window_brightness"] = window_brightness
            self.light_debug["blue_red_ratio"] = ratio

            if window_brightness > brightness * 1.3 and ratio > 1.02:
                self.light_status = "Natural Sunlight"
                self.light_debug["decision"] = "Natural Sunlight (window bright + ratio)"
            else:
                self.light_status = "Artificial Light"
                self.light_debug["decision"] = "Artificial (window/ratio gate)"
        else:
            self.light_status = "Artificial Light"
            self.light_debug["decision"] = "Artificial (no window region)"

        return self.light_status

    # ── Email & Telegram Alerting ──────────────────────────────────────

    def _send_activity_notification(self, action_type, frame=None):
        """Sends an instant alert when someone enters or leaves the room."""
        subject = f"[VISIONCORE] {action_type}: {config.ROOM_NAME}"
        timestamp = __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if action_type == "Activity Appeared":
            body = f"Human activity has been detected in {config.ROOM_NAME}."
        else:
            body = f"The area has been cleared in {config.ROOM_NAME}."
            
        body += f"\nTime: {timestamp}"
        
        # 1. Email Channel
        receiver = getattr(config, 'RECEIVER_EMAIL', '')
        sender = getattr(config, 'SENDER_EMAIL', '')
        password = getattr(config, 'SENDER_PASSWORD', '')

        if receiver and sender and password:
            try:
                from email.mime.multipart import MIMEMultipart
                from email.mime.text import MIMEText
                from email.mime.image import MIMEImage
                import smtplib

                msg = MIMEMultipart()
                msg['Subject'] = subject
                msg['From'] = sender
                msg['To'] = receiver
                msg.attach(MIMEText(body, 'plain'))

                if frame is not None and getattr(frame, 'size', 0) > 0:
                    import cv2
                    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        img_part = MIMEImage(buf.tobytes(), _subtype='jpeg')
                        img_part.add_header('Content-Disposition', 'attachment', filename='snapshot.jpg')
                        msg.attach(img_part)

                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login(sender, password)
                server.send_message(msg)
                server.quit()
                print(f"[ALERT] Instantly dispatched email: {action_type}")
            except Exception as e:
                print(f"[ALERT] Failed to send instant email: {e}")

        # 2. Telegram Channel
        try:
            from logic.telegram_notifier import send_photo, send_message
            if getattr(config, 'TELEGRAM_ENABLED', False):
                tg_caption = f"🚨 <b>{subject}</b>\n{body.replace(chr(10), '<br>')}"
                if frame is not None and getattr(frame, 'size', 0) > 0:
                    import cv2
                    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        send_photo(buf.tobytes(), caption=tg_caption)
                else:
                    send_message(tg_caption)
        except ImportError:
            pass

    def trigger_alert(self, zone_name="default", frame=None):
        # CRITICAL: Copy frame before passing to background threads.
        # The AI thread will overwrite the original array before the
        # Telegram thread gets a chance to encode it as JPEG.
        frame_copy = frame.copy() if frame is not None else None
        print(f"[ALERT] trigger_alert fired — zone={zone_name}, has_frame={frame_copy is not None}")
        threading.Thread(target=lambda: self._send_email_alert(zone_name, frame=frame_copy), daemon=True).start()
        threading.Thread(target=lambda: self._send_telegram_alert(zone_name, frame=frame_copy), daemon=True).start()

    def _send_email_alert(self, zone_name="default", frame=None):
        receiver = config.RECEIVER_EMAIL
        if not receiver or not config.SENDER_EMAIL or not config.SENDER_PASSWORD:
            print("[EMAIL] Skipped — email credentials not configured")
            return
        subject = f"ENERGY ALERT: {config.ROOM_NAME} [{zone_name}]"
        body = (f"The Artificial Lights are ON in {config.ROOM_NAME} (Zone: {zone_name}), "
                f"but no human presence has been detected for over {config.ALERT_DELAY_SECONDS} seconds."
                f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"\n[ALERT!!!] {subject}")
        print(f"[ALERT] Sending to: {receiver}")

        try:
            if frame is not None and getattr(frame, 'size', 0) > 0:
                # Send email with screenshot attachment
                msg = MIMEMultipart()
                msg['Subject'] = subject
                msg['From'] = config.SENDER_EMAIL
                msg['To'] = receiver
                msg.attach(MIMEText(body, 'plain'))
                ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    img_part = MIMEImage(buf.tobytes(), _subtype='jpeg')
                    img_part.add_header('Content-Disposition', 'attachment', filename='evidence.jpg')
                    msg.attach(img_part)
            else:
                msg = MIMEText(body)
                msg['Subject'] = subject
                msg['From'] = config.SENDER_EMAIL
                msg['To'] = receiver

            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(config.SENDER_EMAIL, config.SENDER_PASSWORD)
                server.send_message(msg)
            print(f"[SUCCESS] Email Alert Sent to {receiver}!")
        except Exception as e:
            print(f"[ERROR] Email Failed: {e}")

    def send_test_email(self):
        receiver = config.RECEIVER_EMAIL
        subject = f"[TEST] VisionCore Lab Monitor - {config.ROOM_NAME}"
        body = (f"This is a test email from VisionCore Lab Monitor.\n"
                f"Room: {config.ROOM_NAME}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"If you received this, your alert system is working correctly!")
        print(f"[TEST] Sending test email to: {receiver}")
        try:
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = config.SENDER_EMAIL
            msg['To'] = receiver
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(config.SENDER_EMAIL, config.SENDER_PASSWORD)
                server.send_message(msg)
            print(f"[SUCCESS] Test email sent to {receiver}!")
            return True, f"Test email sent to {receiver}"
        except Exception as e:
            print(f"[ERROR] Test email failed: {e}")
            return False, str(e)

    def _send_telegram_alert(self, zone_name="default", frame=None):
        """Send Telegram alert with optional live frame snapshot."""
        print(f"[TELEGRAM] _send_telegram_alert called — zone={zone_name}, has_frame={frame is not None}")
        if not getattr(config, "TELEGRAM_ENABLED", False):
            print("[TELEGRAM] Skipped — TELEGRAM_ENABLED is False")
            return
        token = getattr(config, "TELEGRAM_BOT_TOKEN", "")
        chat_id = getattr(config, "TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            print(f"[TELEGRAM] Skipped — token={'SET' if token else 'EMPTY'}, chat_id={'SET' if chat_id else 'EMPTY'}")
            return
        try:
            from logic.telegram_notifier import send_alert_with_snapshot, send_message
        except ImportError:
            print("[TELEGRAM] telegram_notifier module not available")
            return

        # Compute waste metrics for the caption
        zone_data = self.zones_state.get(zone_name)
        if zone_data is None:
            # Try matching by zone name
            for zid, zs in self.zones_state.items():
                if zs.get("name") == zone_name:
                    zone_data = zs
                    break
        duration = 0.0
        if zone_data:
            duration = time.time() - zone_data.get("last_seen_time", time.time())
        money = (duration / 3600.0) * (
            getattr(config, "BULB_WATTAGE", 200) / 1000.0
        ) * getattr(config, "ELECTRICITY_RATE", 8.0)

        if frame is not None:
            send_alert_with_snapshot(
                frame_bgr=frame,
                room=config.ROOM_NAME,
                zone=zone_name,
                duration=duration,
                money=money,
            )
        else:
            # Fallback: try to use saved snapshot file
            snapshot_path = getattr(self, "_latest_snapshot_path", None)
            if snapshot_path and os.path.exists(snapshot_path):
                try:
                    with open(snapshot_path, "rb") as f:
                        from logic.telegram_notifier import send_photo
                        caption = (
                            f"⚠️ <b>Energy Waste Alert</b>\n"
                            f"🏢 <b>Room:</b> {config.ROOM_NAME}\n"
                            f"📍 <b>Zone:</b> {zone_name}\n"
                            f"⏱ <b>Duration:</b> {round(duration, 1)}s\n"
                            f"💸 <b>Cost Wasted:</b> ₹{round(money, 2)}"
                        )
                        send_photo(f.read(), caption)
                except Exception:
                    send_message(
                        f"⚠️ <b>Energy Waste Alert</b>\n"
                        f"🏢 Room: {config.ROOM_NAME} | Zone: {zone_name}\n"
                        f"⏱ Duration: {round(duration, 1)}s | 💸 ₹{round(money, 2)}"
                    )
            else:
                send_message(
                    f"⚠️ <b>Energy Waste Alert</b>\n"
                    f"🏢 Room: {config.ROOM_NAME} | Zone: {zone_name}\n"
                    f"⏱ Duration: {round(duration, 1)}s | 💸 ₹{round(money, 2)}"
                )

    def log_energy_waste(self, duration, frame=None, zone_name="default"):
        # Write to the canonical reports folder (stable across different working dirs).
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ghost/
        reports_dir = os.path.join(base_dir, 'reports')
        evidence_dir = os.path.join(base_dir, 'evidence')
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(evidence_dir, exist_ok=True)
        file_path = os.path.join(reports_dir, 'energy_audit.csv')

        # Cost = (duration hours) × (wattage kW) × (rate INR/kWh)
        money_wasted = (duration / 3600.0) * (config.BULB_WATTAGE / 1000.0) * config.ELECTRICITY_RATE
        money_wasted = round(money_wasted, 2)
        event_ts = datetime.now()
        snapshot_path = None
        if frame is not None:
            try:
                snapshot_name = f"waste_{event_ts.strftime('%Y%m%d_%H%M%S')}.jpg"
                snapshot_path = os.path.join(evidence_dir, snapshot_name)
                cv2.imwrite(snapshot_path, frame)
                self._latest_snapshot_path = snapshot_path
            except Exception:
                snapshot_path = None

        # Insert to Supabase Postgres (or SQLite fallback) directly. No CSV native write.
        db.add_waste_event(
            timestamp=event_ts,
            room=config.ROOM_NAME,
            duration_seconds=round(duration, 2),
            status="ALERT_SENT",
            money_wasted=money_wasted,
            snapshot_path=snapshot_path,
            zone_name=zone_name,
        )

    # ── Core Detection Pipeline ───────────────────────────────────────

    def process_frame(self, frame):
        """Run primary tracker + feed verifier. Returns raw frame (no drawing)."""
        t_start = time.time()
        h, w = frame.shape[:2]
        infer_w, infer_h = self._infer_size
        scale_x = w / infer_w
        scale_y = h / infer_h

        # ── Resize for inference ──
        small = cv2.resize(frame, (infer_w, infer_h))

        # ── Primary: YOLOv8n with ByteTrack ──
        try:
            results = self.model_primary.track(
                small, conf=0.25, iou=0.5, verbose=False,
                half=self._use_half, device=self._device,
                persist=True, tracker="bytetrack.yaml",
                classes=[0]  # only track persons
            )
        except Exception:
            # Fallback to plain detect if tracking fails
            results = self.model_primary(
                small, conf=0.25, iou=0.5, verbose=False,
                half=self._use_half, device=self._device,
                classes=[0]
            )

        # ── Extract tracked humans ──
        tracked_humans = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:
                    conf = float(box.conf[0])
                    coords = box.xyxy[0].cpu().numpy().tolist()
                    track_id = int(box.id[0]) if box.id is not None else -1
                    tracked_humans.append(coords + [conf, track_id])

        # ── Get verifier-confirmed humans ──
        with self._verifier_lock:
            verified = list(self._verifier_confirmed)

        # ── Consensus: merge tracker + verifier ──
        confirmed = []
        for th in tracked_humans:
            box_t = th[:4]
            conf_t = th[4]
            tid = th[5]

            # High confidence from tracker alone = accept
            if conf_t >= config.PERSON_ACCEPT_CONF_MIN:
                confirmed.append(th)
                continue

            # Otherwise require verifier agreement
            for vh in verified:
                if _compute_iou(box_t, vh[:4]) >= config.VERIFIER_IOU_MIN:
                    avg_conf = (conf_t + vh[4]) / 2.0
                    confirmed.append(box_t + [avg_conf, tid])
                    break

        # ── Aspect ratio filter (reject non-human shapes) ──
        valid_humans = []
        for human in confirmed:
            bx1, by1, bx2, by2 = human[:4]
            box_w = (bx2 - bx1) * scale_x
            box_h = (by2 - by1) * scale_y
            aspect = box_h / (box_w + 1e-6)
            box_area = box_w * box_h
            frame_area = float(h * w)
            if (
                aspect >= config.PERSON_ASPECT_MIN
                and box_area >= frame_area * config.PERSON_MIN_BOX_AREA_RATIO
            ):
                valid_humans.append(human)

        # Confidence badge uses the currently accepted persons (after filtering).
        if valid_humans:
            self.detection_confidence = sum(h[4] for h in valid_humans) / len(valid_humans)
        else:
            self.detection_confidence = None

        # ── Also extract non-human objects for overlay ──
        all_objects = []
        try:
            # Run a separate full-class detect every few frames
            results_full = self.model_primary(
                small, conf=0.3, iou=0.5, verbose=False,
                half=self._use_half, device=self._device
            )
            for r in results_full:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls != 0:
                        conf = float(box.conf[0])
                        coords = box.xyxy[0].cpu().numpy().tolist()
                        name = self.model_names[cls]
                        all_objects.append({'coords': coords, 'conf': conf, 'name': name})
        except Exception:
            pass

        # ── Feed verifier in background ──
        with self._verifier_lock:
            if not self._verifier_busy:
                self._verifier_frame = small.copy()

        # ── Update overlay data (thread-safe) ──
        with self._overlay_lock:
            self._overlay_humans = valid_humans
            self._overlay_objects = all_objects
            self._overlay_scale = (scale_x, scale_y)

        # ── Temporal count smoothing (majority vote over last 10 frames) ──
        raw_count = len(valid_humans)
        self._count_history.append(raw_count)
        if len(self._count_history) >= 3:
            # Use median for stability
            sorted_counts = sorted(self._count_history)
            mid = len(sorted_counts) // 2
            self.person_count = sorted_counts[mid]
        else:
            self.person_count = raw_count

        # ── Multi-Zone Tracking & Alerting ──
        light = self.analyze_light(frame)
        current_time = time.time()

        # ── Edge-Triggered Activity Notifications ──
        if self._previous_person_count == 0 and self.person_count > 0:
            print(f"[EDGE] Activity Detected: 0 -> {self.person_count}")
            # Ensure frame copy to prevent OpenCV race condition
            frame_c = frame.copy() if frame is not None else None
            threading.Thread(target=lambda: self._send_activity_notification("Activity Appeared", frame_c), daemon=True).start()
        elif self._previous_person_count > 0 and self.person_count == 0:
            print(f"[EDGE] Area Cleared: {self._previous_person_count} -> 0")
            frame_c = frame.copy() if frame is not None else None
            threading.Thread(target=lambda: self._send_activity_notification("Area Cleared", frame_c), daemon=True).start()
            
        self._previous_person_count = self.person_count

        zone_counts = {zid: 0 for zid in self.zones_state.keys()}
        for human in valid_humans:
            bx1, by1, bx2, by2 = human[:4]
            cx = (bx1 + bx2) / 2.0 / infer_w
            cy = (by1 + by2) / 2.0 / infer_h
            for zid, zstate in self.zones_state.items():
                zx1, zy1, zx2, zy2 = zstate["bbox"]
                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                    zone_counts[zid] += 1

        self.is_energy_wasted = False
        self.alert_sent = False

        for zid, zstate in self.zones_state.items():
            count = zone_counts[zid]
            zstate["person_count"] = count
            if count > 0:
                zstate["last_seen_time"] = current_time
                zstate["is_energy_wasted"] = False
                zstate["alert_sent"] = False
            else:
                empty_duration = current_time - zstate["last_seen_time"]
                if light == "Artificial Light" and empty_duration >= config.ALERT_DELAY_SECONDS:
                    zstate["is_energy_wasted"] = True
                    self.is_energy_wasted = True
                    if not zstate["alert_sent"]:
                        zstate["alert_sent"] = True
                        self.alert_sent = True
                        self.trigger_alert(zone_name=zstate["name"], frame=frame)
                        self.log_energy_waste(empty_duration, frame=frame, zone_name=zstate["name"])
                else:
                    zstate["is_energy_wasted"] = False

        # ── Auto-Off Logic ──────────────────────────────────────────────
        if self.person_count > 0:
            self._global_last_seen = current_time
            # Person returned → auto-on if was off
            if self.auto_off_active:
                self.auto_off_active = False
                self._auto_off_notified = False
                print(f"[AUTO-ON] Activity resumed in {config.ROOM_NAME}")
                # Send "room re-activated" notification
                threading.Thread(
                    target=self._send_auto_on_notification, daemon=True
                ).start()
        else:
            auto_off_delay = getattr(config, "AUTO_OFF_DELAY_MINUTES", 0)
            if auto_off_delay > 0:
                empty_global = current_time - self._global_last_seen
                if empty_global >= auto_off_delay * 60:
                    self.auto_off_active = True
                    if not self._auto_off_notified:
                        self._auto_off_notified = True
                        print(f"[AUTO-OFF] No activity for {auto_off_delay}min in {config.ROOM_NAME}")
                        frame_copy = frame.copy() if frame is not None else None
                        threading.Thread(
                            target=lambda: self._send_auto_off_notification(frame_copy),
                            daemon=True,
                        ).start()

        # ── Track AI FPS ──
        t_elapsed = time.time() - t_start
        self._fps_ring.append(1.0 / max(t_elapsed, 1e-6))
        self._ai_fps = round(sum(self._fps_ring) / len(self._fps_ring), 1)

        # Snapshot light telemetry for the video overlay thread.
        with self._overlay_lock:
            self._overlay_luminance = self.luminance
            self._overlay_light_status = self.light_status

        return frame

    # ── Auto-Off / Auto-On Notifications ──────────────────────────────

    def _send_auto_off_notification(self, frame=None):
        """Notify via email + Telegram that the room has been auto-powered-down."""
        text = (
            f"🔴 <b>Auto Power-Down</b>\n"
            f"🏢 <b>Room:</b> {config.ROOM_NAME}\n"
            f"⏱ No activity for {getattr(config, 'AUTO_OFF_DELAY_MINUTES', 10)} minutes\n"
            f"💡 Lights should be turned OFF\n"
            f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        # Telegram
        try:
            from logic.telegram_notifier import send_alert_with_snapshot, send_message
            if frame is not None and getattr(frame, 'size', 0) > 0:
                send_alert_with_snapshot(
                    frame_bgr=frame, room=config.ROOM_NAME,
                    zone="ALL", duration=getattr(config, 'AUTO_OFF_DELAY_MINUTES', 10) * 60,
                    money=0
                )
            else:
                send_message(text)
        except Exception as e:
            print(f"[AUTO-OFF] Telegram failed: {e}")
        # Email
        try:
            self._send_email_alert(zone_name="AUTO-OFF", frame=frame)
        except Exception as e:
            print(f"[AUTO-OFF] Email failed: {e}")

    def _send_auto_on_notification(self):
        """Notify that the room has been re-activated (person returned)."""
        text = (
            f"🟢 <b>Room Re-Activated</b>\n"
            f"🏢 <b>Room:</b> {config.ROOM_NAME}\n"
            f"👤 Activity detected — system monitoring resumed\n"
            f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        try:
            from logic.telegram_notifier import send_message
            send_message(text)
        except Exception:
            pass

    # ── Periodic Status Log ───────────────────────────────────────────

    def send_periodic_log(self, frame=None):
        """Send a periodic status summary to email + Telegram with screenshot.
        Called by the background thread in app.py."""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        status = "OCCUPIED" if self.person_count > 0 else "EMPTY"
        light = self.light_status
        auto_off = "YES" if self.auto_off_active else "NO"

        text = (
            f"📊 <b>Periodic Status Log</b>\n"
            f"🏢 <b>Room:</b> {config.ROOM_NAME}\n"
            f"👤 <b>Status:</b> {status} ({self.person_count} persons)\n"
            f"💡 <b>Light:</b> {light}\n"
            f"🔴 <b>Auto-Off:</b> {auto_off}\n"
            f"⚡ <b>Waste Active:</b> {'YES' if self.is_energy_wasted else 'NO'}\n"
            f"🕐 <b>Time:</b> {now}"
        )

        print(f"[PERIODIC LOG] Sending status log at {now}")

        # Telegram
        try:
            from logic.telegram_notifier import send_alert_with_snapshot, send_message
            if frame is not None and getattr(frame, 'size', 0) > 0:
                import cv2
                ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    from logic.telegram_notifier import send_photo
                    send_photo(buf.tobytes(), text)
                else:
                    send_message(text)
            else:
                send_message(text)
        except Exception as e:
            print(f"[PERIODIC LOG] Telegram failed: {e}")

        # Email
        try:
            receiver = config.RECEIVER_EMAIL
            if receiver and config.SENDER_EMAIL and config.SENDER_PASSWORD:
                subject = f"[STATUS LOG] {config.ROOM_NAME} — {status}"
                body = text.replace('<b>', '').replace('</b>', '')

                if frame is not None and getattr(frame, 'size', 0) > 0:
                    msg = MIMEMultipart()
                    msg['Subject'] = subject
                    msg['From'] = config.SENDER_EMAIL
                    msg['To'] = receiver
                    msg.attach(MIMEText(body, 'plain'))
                    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        img_part = MIMEImage(buf.tobytes(), _subtype='jpeg')
                        img_part.add_header('Content-Disposition', 'attachment', filename='status.jpg')
                        msg.attach(img_part)
                else:
                    msg = MIMEText(body)
                    msg['Subject'] = subject
                    msg['From'] = config.SENDER_EMAIL
                    msg['To'] = receiver

                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                    server.login(config.SENDER_EMAIL, config.SENDER_PASSWORD)
                    server.send_message(msg)
                print(f"[PERIODIC LOG] Email sent to {receiver}")
        except Exception as e:
            print(f"[PERIODIC LOG] Email failed: {e}")

    # ── Draw Overlay (called by stream thread — never blocks AI) ─────

    def draw_overlay(self, frame):
        """Draw detection boxes onto a raw frame. Called by the stream, not AI."""
        with self._overlay_lock:
            humans = list(self._overlay_humans)
            objects = list(self._overlay_objects)
            scale_x, scale_y = self._overlay_scale
            lum = self._overlay_luminance
            light_status = self._overlay_light_status

        # ── Draw non-human objects ──
        for obj in objects:
            bx1, by1, bx2, by2 = obj['coords']
            x1o = int(bx1 * scale_x)
            y1o = int(by1 * scale_y)
            x2o = int(bx2 * scale_x)
            y2o = int(by2 * scale_y)
            cv2.rectangle(frame, (x1o, y1o), (x2o, y2o), (255, 165, 0), 1)
            cv2.putText(frame, f"{obj['name']} {obj['conf']:.0%}",
                        (x1o, y1o - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)

        # ── Draw humans with tracking IDs ──
        for human in humans:
            bx1, by1, bx2, by2 = human[:4]
            conf = human[4]
            tid = int(human[5]) if len(human) > 5 else -1

            x1h = int(bx1 * scale_x)
            y1h = int(by1 * scale_y)
            x2h = int(bx2 * scale_x)
            y2h = int(by2 * scale_y)

            # Green box with glow
            cv2.rectangle(frame, (x1h, y1h), (x2h, y2h), (0, 255, 0), 2)

            label = f"HUMAN {conf:.0%}"
            if tid >= 0:
                label = f"ID:{tid} {conf:.0%}"
            (lw, lh_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1h, y1h - lh_t - 10), (x1h + lw, y1h), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1h, y1h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # ── Draw luminance + light classification on the frame ──
        # This makes it obvious (even without JS) that the AI is actively computing luminance.
        try:
            if lum is None:
                lum_int = None
            else:
                lum_int = int(max(0, min(255, round(float(lum)))))

            x = 14
            y = 28
            panel_w = 260
            panel_h = 56
            cv2.rectangle(frame, (x - 6, y - 22), (x - 6 + panel_w, y - 22 + panel_h), (0, 0, 0), -1)
            cv2.rectangle(frame, (x - 6, y - 22), (x - 6 + panel_w, y - 22 + panel_h), (148, 163, 184), 1)

            top = "LUMINANCE"
            if lum_int is None:
                lum_str = "--"
            else:
                lum_str = str(lum_int)
            cv2.putText(frame, f"{top}: {lum_str}/255", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, str(light_status), (x, y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        except Exception:
            # Never fail overlay drawing.
            pass

        # ── Draw Multi-Zone overlays ──
        try:
            zones_map = getattr(config, 'ZONES_MAP', {})
            h, w = frame.shape[:2]
            zone_colors = [
                (200, 120, 255),  # purple
                (120, 220, 255),  # cyan
                (255, 200, 100),  # gold
                (100, 255, 160),  # mint
            ]
            for idx, (zid, zdata) in enumerate(zones_map.items()):
                bbox = zdata.get('bbox', [0, 0, 1, 1])
                zname = zdata.get('name', zid)
                zx1 = int(bbox[0] * w)
                zy1 = int(bbox[1] * h)
                zx2 = int(bbox[2] * w)
                zy2 = int(bbox[3] * h)
                color = zone_colors[idx % len(zone_colors)]
                cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), color, 1)
                # Label background
                label = zname.upper()
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (zx1, zy1), (zx1 + lw + 8, zy1 + lh + 8), color, -1)
                cv2.putText(frame, label, (zx1 + 4, zy1 + lh + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        except Exception:
            pass

        return frame
