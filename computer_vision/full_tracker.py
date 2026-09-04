"""
YOLOv8 Pose Tracker - Clean & Optimized for Raspberry Pi
"""

import cv2
import numpy as np
import time
from collections import deque
import os
from ultralytics import YOLO

# ========================= CONFIG =========================
PIXELS_PER_METER = 130.0
MOVING_THRESHOLD = 0.35
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# =======================================================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

model = None
use_video = False
current_video_path = None

trajectory = deque(maxlen=500)
center_history = deque(maxlen=8)

total_distance = 0.0
moving_time = 0.0
max_speed = 0.0
prev_center = None
prev_time = time.time()
frame_count = 0
last_results = None

def reset_stats():
    global total_distance, moving_time, max_speed, prev_center, frame_count, last_results
    total_distance = moving_time = max_speed = 0.0
    trajectory.clear()
    center_history.clear()
    prev_center = None
    frame_count = 0
    last_results = None
    print("Stats reset")

def pick_video_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All Files", "*.*")]
        )
        root.destroy()
        return path
    except Exception as e:
        print(f"File dialog failed: {e}")
        return None

def show_end_screen(cap):
    global use_video, current_video_path

    print("\n========== VIDEO COMPLETE ==========")
    avg_speed = total_distance / moving_time if moving_time > 0 else 0.0
    print(f"  Total Distance : {total_distance:.2f} m")
    print(f"  Moving Time    : {moving_time:.1f} s")
    print(f"  Avg Speed      : {avg_speed:.2f} m/s")
    print(f"  Max Speed      : {max_speed:.2f} m/s")
    print("=====================================")

    stats_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    overlay = stats_frame.copy()
    cv2.rectangle(overlay, (50, 80), (590, 440), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, stats_frame, 0.4, 0, stats_frame)

    cv2.putText(stats_frame, "VIDEO COMPLETE",                       (150, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2,  (0, 255, 255), 3)
    cv2.putText(stats_frame, f"Distance   : {total_distance:.2f} m", (80, 190),  cv2.FONT_HERSHEY_SIMPLEX, 0.9,  (255, 255, 255), 2)
    cv2.putText(stats_frame, f"Moving Time: {moving_time:.1f} s",    (80, 235),  cv2.FONT_HERSHEY_SIMPLEX, 0.9,  (255, 255, 255), 2)
    cv2.putText(stats_frame, f"Avg Speed  : {avg_speed:.2f} m/s",    (80, 280),  cv2.FONT_HERSHEY_SIMPLEX, 0.9,  (255, 255, 255), 2)
    cv2.putText(stats_frame, f"Max Speed  : {max_speed:.2f} m/s",    (80, 325),  cv2.FONT_HERSHEY_SIMPLEX, 0.9,  (255, 255, 255), 2)
    cv2.putText(stats_frame, "R=Replay  N=New file  W=Webcam  Q=Quit", (55, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Pose Tracker", stats_frame)

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord('r'):
            reset_stats()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            print("Replaying video...")
            return cap
        elif key == ord('n'):
            reset_stats()
            cap.release()
            file_path = pick_video_file()
            if file_path and os.path.exists(file_path):
                cap = cv2.VideoCapture(file_path)
                current_video_path = file_path
                print(f"Loaded: {os.path.basename(file_path)}")
            else:
                use_video = False
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                current_video_path = None
                print("No file selected — back to webcam")
            return cap
        elif key == ord('w'):
            reset_stats()
            cap.release()
            use_video = False
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            current_video_path = None
            print("Switched to Webcam")
            return cap
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("Tracker stopped.")
            exit()

print("Press 'm' → Switch mode | 'r' → Reset | 'q' → Quit")
print("Loading... window will appear first, model loads after.")

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        if use_video:
            cap = show_end_screen(cap)
        else:
            time.sleep(0.05)
        continue

    # === Lazy load YOLO after first frame ===
    if model is None:
        cv2.putText(frame, "Loading model...", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("YOLOv8 Pose Tracker", frame)
        cv2.waitKey(1)
        model = YOLO("yolov8n-pose.pt")
        model.overrides['imgsz'] = 320
        print("Model loaded.")
        prev_time = time.time()
        continue

    # === dt calculation ===
    if use_video:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 25.0
        dt = 1.0 / video_fps
    else:
        now = time.time()
        dt = min(now - prev_time, 0.2)
        prev_time = now

    # === Frame skipping ===
    frame_count += 1
    if frame_count % 2 == 0:
        last_results = model(frame, verbose=False, conf=0.4)
    results = last_results if last_results else []

    person_detected = False

    if results and len(results) > 0 and results[0].keypoints is not None:
        kp_data = results[0].keypoints.xy.cpu().numpy()
        if len(kp_data) > 0:
            kp = kp_data[0]  # First person

            left_hip  = kp[11] if len(kp) > 11 else None
            right_hip = kp[12] if len(kp) > 12 else None
            left_sho  = kp[5]  if len(kp) > 5  else None
            right_sho = kp[6]  if len(kp) > 6  else None

            if left_hip is not None and right_hip is not None and (left_hip[0] > 0 or right_hip[0] > 0):
                cx = int((left_hip[0] + right_hip[0]) / 2)
                cy = int((left_hip[1] + right_hip[1]) / 2)
                person_detected = True
            elif left_sho is not None and right_sho is not None and (left_sho[0] > 0 or right_sho[0] > 0):
                cx = int((left_sho[0] + right_sho[0]) / 2)
                cy = int((left_sho[1] + right_sho[1]) / 2)
                person_detected = True

            if person_detected:
                center_history.append((cx, cy))
                if len(center_history) >= 5:
                    cx = int(np.mean([p[0] for p in center_history]))
                    cy = int(np.mean([p[1] for p in center_history]))

                for x, y in kp:
                    if x > 0 and y > 0:
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

                trajectory.append((cx, cy))
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                if prev_center is not None:
                    dx = cx - prev_center[0]
                    dy = cy - prev_center[1]
                    dist_pixels = np.sqrt(dx * dx + dy * dy)
                    dist_meters = dist_pixels / PIXELS_PER_METER
                    total_distance += dist_meters
                    speed = dist_meters / dt if dt > 0 else 0
                    if speed > max_speed:
                        max_speed = speed
                    if speed > MOVING_THRESHOLD:
                        moving_time += dt

                prev_center = (cx, cy)

    for i in range(1, len(trajectory)):
        cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 0), 2)

    avg_speed = total_distance / moving_time if moving_time > 0 else 0.0

    mode_text = f"VIDEO: {os.path.basename(current_video_path)}" if use_video and current_video_path else "LIVE WEBCAM"
    cv2.putText(frame, mode_text,                               (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.6,  (255, 255, 0), 2)
    cv2.putText(frame, f"Dist: {total_distance:.2f} m",        (10, 55),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv2.putText(frame, f"Time: {moving_time:.1f} s",           (10, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv2.putText(frame, f"Avg : {avg_speed:.2f} m/s",           (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv2.putText(frame, f"Max : {max_speed:.2f} m/s",           (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    if not person_detected:
        cv2.putText(frame, "NO PERSON DETECTED", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("YOLOv8 Pose Tracker", frame)

    key = cv2.waitKey(1 if use_video else 10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        reset_stats()
    elif key == ord('m'):
        cap.release()
        file_path = pick_video_file()
        if file_path and os.path.exists(file_path):
            reset_stats()
            use_video = True
            cap = cv2.VideoCapture(file_path)
            current_video_path = file_path
            print(f"Loaded: {os.path.basename(file_path)}")
        else:
            use_video = False
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            current_video_path = None
            print("No file selected — staying on webcam")

cap.release()
cv2.destroyAllWindows()
print("Tracker stopped.")