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
import torch
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import config
except ImportError:
    from .. import config


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
        self.last_seen_time = time.time()
        self.light_status = "Dark"
        self.is_energy_wasted = False
        self.alert_sent = False
        # Average confidence (0.0-1.0) from the primary tracker for current detections.
        # Used by the /health and /status endpoints.
        self.detection_confidence = None

        # ── Inference settings ──
        self._infer_size = (320, 320) if self._device == 'cuda' else (256, 256)

        # ── Performance tracking ──
        self._ai_fps = 0.0
        self._fps_ring = deque(maxlen=30)

        # ── Temporal smoothing for stable count ──
        self._count_history = deque(maxlen=10)

        # ── Overlay data (shared with stream thread) ──
        self._overlay_lock = threading.Lock()
        self._overlay_humans = []       # [(x1,y1,x2,y2,conf,track_id), ...]
        self._overlay_objects = []      # [{'coords':..., 'name':..., 'conf':...}, ...]
        self._overlay_scale = (1.0, 1.0)

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
        if brightness < config.BRIGHTNESS_THRESHOLD:
            self.light_status = "Dark"
            return self.light_status

        if self._is_night_time():
            self.light_status = "Artificial Light"
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

            if window_brightness > brightness * 1.3 and ratio > 1.02:
                self.light_status = "Natural Sunlight"
            else:
                self.light_status = "Artificial Light"
        else:
            self.light_status = "Artificial Light"

        return self.light_status

    # ── Email Alerting ────────────────────────────────────────────────

    def trigger_alert(self):
        if not self.alert_sent:
            self.alert_sent = True
            threading.Thread(target=self._send_email_alert, daemon=True).start()

    def _send_email_alert(self):
        receiver = config.RECEIVER_EMAIL
        subject = f"ENERGY ALERT: {config.ROOM_NAME}"
        body = (f"The Artificial Lights are ON in {config.ROOM_NAME}, "
                f"but no human presence has been detected for over {config.ALERT_DELAY_SECONDS} seconds.")

        print(f"\n[ALERT!!!] {subject}")
        print(f"[ALERT] Sending to: {receiver}")

        try:
            msg = MIMEText(body + f"\nTime: {datetime.now().strftime('%H:%M:%S')}")
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

    def log_energy_waste(self, duration):
        # Write to the canonical reports folder (stable across different working dirs).
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ghost/
        reports_dir = os.path.join(base_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        file_path = os.path.join(reports_dir, 'energy_audit.csv')

        # Cost = (duration hours) × (wattage kW) × (rate INR/kWh)
        money_wasted = (duration / 3600.0) * (config.BULB_WATTAGE / 1000.0) * config.ELECTRICITY_RATE
        money_wasted = round(money_wasted, 2)

        with open(file_path, mode='a', newline='') as f:
            import csv
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(['Timestamp', 'Room', 'Duration_Seconds', 'Status', 'Money_Wasted'])
            writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                             config.ROOM_NAME,
                             round(duration, 2),
                             'ALERT_SENT',
                             money_wasted])

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

        # Compute average confidence from the primary tracker for UI telemetry.
        if tracked_humans:
            self.detection_confidence = sum(th[4] for th in tracked_humans) / len(tracked_humans)
        else:
            self.detection_confidence = None

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
            if conf_t >= 0.50:
                confirmed.append(th)
                continue

            # Otherwise require verifier agreement
            for vh in verified:
                if _compute_iou(box_t, vh[:4]) >= 0.3:
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
            if aspect >= 0.7:
                valid_humans.append(human)

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

        # ── Light analysis & alerting ──
        light = self.analyze_light(frame)
        current_time = time.time()

        if self.person_count > 0:
            self.last_seen_time = current_time
            self.is_energy_wasted = False
            self.alert_sent = False
        else:
            empty_duration = current_time - self.last_seen_time
            # ONLY waste if Artificial Light is ON and room is Empty
            if light == "Artificial Light" and empty_duration >= config.ALERT_DELAY_SECONDS:
                self.is_energy_wasted = True
                if not self.alert_sent:
                    self.trigger_alert()
                    self.log_energy_waste(empty_duration)
            else:
                # If it is Dark or Natural or just started being empty, no waste.
                self.is_energy_wasted = False

        # ── Track AI FPS ──
        t_elapsed = time.time() - t_start
        self._fps_ring.append(1.0 / max(t_elapsed, 1e-6))
        self._ai_fps = round(sum(self._fps_ring) / len(self._fps_ring), 1)

        return frame

    # ── Draw Overlay (called by stream thread — never blocks AI) ─────

    def draw_overlay(self, frame):
        """Draw detection boxes onto a raw frame. Called by the stream, not AI."""
        with self._overlay_lock:
            humans = list(self._overlay_humans)
            objects = list(self._overlay_objects)
            scale_x, scale_y = self._overlay_scale

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

        return frame
