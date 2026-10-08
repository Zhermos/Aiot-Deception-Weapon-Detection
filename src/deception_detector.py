import cv2
import mediapipe as mp
import numpy as np
import math
import time
import base64
import paho.mqtt.client as mqtt
from collections import deque

# ---------------- SETTINGS / THRESHOLDS ----------------
USE_MQTT = True
MQTT_BROKER = "mqtt-ajoy.ddns.net"
MQTT_PORT = 1883
MQTT_USERNAME = "AIoT"
MQTT_PASSWORD = "codingthailand"
MQTT_TOPIC_RISK = "/TUP/Tuppong/RISK"
MQTT_TOPIC_SCORE = "/TUP/Tuppong/SCORE"
MQTT_TOPIC_FREEZ = "/TUP/Tuppong/FREEZ"

BLINK_EAR_THRESHOLD = 0.21
BLINK_CONSEC_FRAMES = 2
BLINK_WINDOW_SECONDS = 10
INCREASED_BLINK_RATE = 5.0

GAZE_OFFSET_THRESHOLD = 0.22
HEAD_YAW_THRESHOLD = 15.0

FIDGET_WINDOW = 15.0
FIDGET_THRESHOLD = 0.01
FREEZE_THRESHOLD = 0.1

WEIGHT_BLINK = 0.35
WEIGHT_GAZE = 0.25
WEIGHT_BODY = 0.25
WEIGHT_FREEZE = 0.09

LOW_THRESHOLD = 25
MEDIUM_THRESHOLD = 50
HIGH_THRESHOLD = 75

last_sent_time = 0
SEND_DELAY = 10

# ---------------- MQTT ----------------
if USE_MQTT:
    mqtt_client = mqtt.Client()
    mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        mqtt_client.loop_start()
        print("Connected to MQTT")
    except:
        print("MQTT connect failed")
        mqtt_client = None
else:
    mqtt_client = None

# ---------------- Mediapipe ----------------
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)
pose = mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

MODEL_POINTS = np.array([
    (0, 0, 0),
    (0, -63, -12),
    (-43, 32, -26),
    (43, 32, -26),
    (-28, -28, -24),
    (28, -28, -24)
], dtype=np.float64)

# ---------------- UTIL ----------------
def eye_aspect_ratio(pts):
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    if C == 0: return 0
    return (A + B) / (2.0 * C)

def iris_offset(eye_pts, iris_pt):
    left = eye_pts[0]
    right = eye_pts[3]
    rng = np.linalg.norm(right - left)
    if rng == 0: return 0
    center_x = (left[0] + right[0]) / 2.0
    return (iris_pt[0] - center_x) / rng

def safe_head_pose(lm, w, h):
    try:
        idx = [1,152,263,33,287,57]
        pts = [(lm[i].x*w, lm[i].y*h) for i in idx]
        image_points = np.array(pts, dtype=np.float64)
        focal = w
        cam = np.array([[focal,0,w/2],
                        [0,focal,h/2],
                        [0,0,1]], float)
        dist = np.zeros((4,1))
        ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points, cam, dist)
        if not ok: return 0,0,0
        rot,_ = cv2.Rodrigues(rvec)
        sy = math.sqrt(rot[0,0]**2 + rot[1,0]**2)
        pitch = math.degrees(math.atan2(rot[2,1], rot[2,2]))
        yaw = math.degrees(math.atan2(-rot[2,0], sy))
        roll = math.degrees(math.atan2(rot[1,0], rot[0,0]))
        return pitch,yaw,roll
    except:
        return 0,0,0

# ---------------- Buffers ----------------
blink_frame = 0
blink_times = deque()
motion_buffer = deque()

# ---------------- Main ----------------
cap = cv2.VideoCapture(0)
print("ระบบเริ่มทำงาน…")

try:
    while True:
        ret, frame = cap.read()
        if not ret: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        t_now = time.time()

        face_res = face_mesh.process(rgb)
        pose_res = pose.process(rgb)

        blink_rate = 0
        gaze_flag = 0
        yaw_flag = 0
        fidget_norm = 0
        freeze_flag = False

        # -------- FACE --------
        if face_res.multi_face_landmarks:
            lm = face_res.multi_face_landmarks[0].landmark

            # ดึง landmark
            L = [33,160,158,133,153,144]
            R = [263,387,385,362,380,373]
            IL = 468
            IR = 473
            left = np.array([[lm[i].x*w, lm[i].y*h] for i in L])
            right = np.array([[lm[i].x*w, lm[i].y*h] for i in R])
            irisL = np.array([lm[IL].x*w, lm[IL].y*h])
            irisR = np.array([lm[IR].x*w, lm[IR].y*h])

            ear_l = eye_aspect_ratio(left)
            ear_r = eye_aspect_ratio(right)
            ear_avg = (ear_l + ear_r) / 2

            # ---- Detect Blink ----
            if ear_avg < BLINK_EAR_THRESHOLD:
                blink_frame += 1
            else:
                if blink_frame >= BLINK_CONSEC_FRAMES:
                    blink_times.append(t_now)
                blink_frame = 0

            # Remove old blinks
            while blink_times and t_now - blink_times[0] > BLINK_WINDOW_SECONDS:
                blink_times.popleft()

            blink_rate = len(blink_times) * (10.0 / BLINK_WINDOW_SECONDS)

            # ---- Gaze ----
            offL = abs(iris_offset(left, irisL))
            offR = abs(iris_offset(right, irisR))
            gaze_flag = int(offL > GAZE_OFFSET_THRESHOLD or offR > GAZE_OFFSET_THRESHOLD)

            pitch, yaw, roll = safe_head_pose(lm, w, h)
            yaw_flag = int(abs(yaw) > HEAD_YAW_THRESHOLD)

            # Landmark draw only for display, not in MQTT image
            display_frame = frame.copy()
            mp_draw.draw_landmarks(
                display_frame, face_res.multi_face_landmarks[0],
                mp_face.FACEMESH_TESSELATION,
                None,
                mp_draw.DrawingSpec((0,120,255),1)
            )

        else:
            display_frame = frame.copy()

        # -------- BODY MOTION --------
        if pose_res.pose_landmarks:
            pl = pose_res.pose_landmarks.landmark
            try:
                nose = np.array([pl[mp_pose.PoseLandmark.NOSE.value].x,
                                 pl[mp_pose.PoseLandmark.NOSE.value].y])
                ls = np.array([pl[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                               pl[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
                rs = np.array([pl[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                               pl[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
            except:
                nose = ls = rs = None

            if nose is not None:
                motion_buffer.append((t_now, nose, ls, rs))
                while motion_buffer and t_now - motion_buffer[0][0] > FIDGET_WINDOW:
                    motion_buffer.popleft()

                if len(motion_buffer) >= 2:
                    speeds = []
                    buf = list(motion_buffer)
                    for i in range(1, len(buf)):
                        dt = buf[i][0] - buf[i-1][0]
                        if dt == 0: continue
                        v1 = np.linalg.norm(buf[i][1] - buf[i-1][1]) / dt
                        v2 = np.linalg.norm(buf[i][2] - buf[i-1][2]) / dt
                        v3 = np.linalg.norm(buf[i][3] - buf[i-1][3]) / dt
                        speeds.append((v1+v2+v3)/3)
                    avg = np.mean(speeds) if speeds else 0
                    fidget_norm = min(1.0, avg / (FIDGET_THRESHOLD*6))
                    freeze_flag = avg < FREEZE_THRESHOLD

        # -------- SCORE --------
        blink_norm = min(1, blink_rate / INCREASED_BLINK_RATE)
        gaze_total = int(gaze_flag or yaw_flag)
        freeze_norm = 1.0 if freeze_flag else 0.0

        score = (
            WEIGHT_BLINK * blink_norm +
            WEIGHT_GAZE * gaze_total +
            WEIGHT_BODY * fidget_norm +
            WEIGHT_FREEZE * freeze_norm
        ) * 100

        # -------- LABEL --------
        if blink_rate >= 21:
            label = "VERY HIGH"; color = (0,0,255)
        elif score >= 90:
            label = "VERY HIGH"; color = (0,0,255)
        elif score >= 75:
            label = "HIGH"; color = (0,100,255)
        elif score >= 50:
            label = "MEDIUM"; color = (0,255,255)
        elif score >= 25:
            label = "LOW"; color = (0,255,0)
        else:
            label = "VERY LOW"; color = (150,255,150)
        
        # -------- SEND IMAGE MQTT (WITHOUT LANDMARKS) --------
        if label == "VERY HIGH" and mqtt_client:
            t_now = time.time()
            if t_now - last_sent_time >= SEND_DELAY:
                _, buffer = cv2.imencode('.jpg', frame)  # send raw frame without landmarks
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                try:
                    mqtt_client.publish("/TUP/Tuppong/ALERT_IMAGE", jpg_as_text)
                    print("ส่งภาพ VERY HIGH ไปยัง MQTT แล้ว")
                    last_sent_time = t_now
                except:
                    print("ส่งภาพ MQTT ล้มเหลว")

        # -------- UI Display --------
        cv2.putText(display_frame, f"EAR L: {ear_l:.3f}", (20,45), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0),2)
        cv2.putText(display_frame, f"EAR R: {ear_r:.3f}", (20,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0),2)
        cv2.putText(display_frame, f"BlinkRate: {blink_rate:.2f}/10s", (20,115), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)
        gd = "center"
        if gaze_flag: gd = "eyes"
        if yaw_flag: gd += " + head"
        cv2.putText(display_frame, f"Gaze: {gd}", (20,155), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,0),2)
        cv2.putText(display_frame, f"Score:{score:.1f} [{label}]", (12, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # -------- MQTT --------
        if mqtt_client:
            try:
                mqtt_client.publish(MQTT_TOPIC_RISK, label)
                mqtt_client.publish(MQTT_TOPIC_SCORE, f"{score:.2f}")
                mqtt_client.publish(MQTT_TOPIC_FREEZ, str(int(freeze_flag)))
            except:
                pass

        cv2.imshow("Deception Detection", display_frame)
        cv2.imshow("RAW Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()