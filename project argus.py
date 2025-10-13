#!/usr/bin/env python
# -- coding: utf-8 --



import cv2
import time
import mediapipe as mp
import math
import logging
import numpy as np

# =============================================================================
# 1. INITIALIZATION AND SETUP
# =============================================================================

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('drowsiness_alerts.log')
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --- General Video Capture Settings ---
FRAME_W = 640
FRAME_H = 480

# --- Detection Constants ---
EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 25
MOUTH_OPEN_THRESH = 25
MOUTH_OPEN_CONSEC_FRAMES = 25
ROLL_THRESH_DEGREES = 20  # Side-to-side tilt
ROLL_CONSEC_FRAMES = 20
PITCH_THRESH_DEGREES = 20 # Up/down tilt
PITCH_CONSEC_FRAMES = 20
FACE_MISSING_SECONDS = 3.0

# --- State counters and alert flags ---
EYE_COUNTER, MOUTH_COUNTER, ROLL_COUNTER, PITCH_COUNTER = 0, 0, 0, 0
EYES_CLOSED_ALERT_TRIGGERED, YAWN_ALERT_TRIGGERED = False, False
ROLL_ALERT_TRIGGERED, PITCH_ALERT_TRIGGERED = False, False
FACE_MISSING_START_TIME, FACE_MISSING_ALERT_TRIGGERED = None, False

# --- Face Mesh Detection (MediaPipe) Setup ---
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Video Capture Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error("Could not open video stream. Make sure a webcam is connected.")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
time.sleep(2)

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def calculate_ear(eye_landmarks, frame_shape):
    # Calculates Eye Aspect Ratio
    p2 = (int(eye_landmarks[159].x * frame_shape[1]), int(eye_landmarks[159].y * frame_shape[0]))
    p6 = (int(eye_landmarks[145].x * frame_shape[1]), int(eye_landmarks[145].y * frame_shape[0]))
    p3 = (int(eye_landmarks[158].x * frame_shape[1]), int(eye_landmarks[158].y * frame_shape[0]))
    p5 = (int(eye_landmarks[153].x * frame_shape[1]), int(eye_landmarks[153].y * frame_shape[0]))
    p1 = (int(eye_landmarks[33].x * frame_shape[1]), int(eye_landmarks[33].y * frame_shape[0]))
    p4 = (int(eye_landmarks[133].x * frame_shape[1]), int(eye_landmarks[133].y * frame_shape[0]))
    ver_dist1 = math.hypot(p2[0] - p6[0], p2[1] - p6[1])
    ver_dist2 = math.hypot(p3[0] - p5[0], p3[1] - p5[1])
    hor_dist = math.hypot(p1[0] - p4[0], p1[1] - p4[1])
    if hor_dist == 0: return 0.3
    return (ver_dist1 + ver_dist2) / (2.0 * hor_dist)

def get_mouth_opening(mouth_landmarks, frame_shape):
    # Calculates vertical mouth opening distance
    upper_lip = (int(mouth_landmarks[13].x * frame_shape[1]), int(mouth_landmarks[13].y * frame_shape[0]))
    lower_lip = (int(mouth_landmarks[14].x * frame_shape[1]), int(mouth_landmarks[14].y * frame_shape[0]))
    return math.hypot(upper_lip[0] - lower_lip[0], upper_lip[1] - lower_lip[1])

def get_head_pose(landmarks, frame_shape):
    """
    Estimates head pose angles (pitch, yaw, roll) using solvePnP.
    Returns:
        tuple: (pitch, yaw, roll) angles in degrees, or None if estimation fails.
    """
    h, w, _ = frame_shape
    face_model_3d = np.array([
        (0.0, 0.0, 0.0),            # Nose tip
        (0.0, -330.0, -65.0),       # Chin
        (-225.0, 170.0, -135.0),    # Left eye left corner
        (225.0, 170.0, -135.0),     # Right eye right corner
        (-150.0, -150.0, -125.0),   # Left Mouth corner
        (150.0, -150.0, -125.0)     # Right mouth corner
    ], dtype=np.float64)
    
    pnp_landmark_ids = [1, 152, 263, 33, 291, 61]
    image_points_2d = np.array([
        (landmarks[id].x * w, landmarks[id].y * h) for id in pnp_landmark_ids
    ], dtype=np.float64)

    camera_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rot_vec, _ = cv2.solvePnP(
        face_model_3d, image_points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success: return None

    rot_mat, _ = cv2.Rodrigues(rot_vec)
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(np.hstack((rot_mat, np.zeros((3,1)))))
    
    pitch, yaw, roll = euler_angles.flatten()[:3]
    return pitch, yaw, roll

# =============================================================================
# 3. MAIN PROCESSING LOOP
# =============================================================================
logger.info("Starting video processing loop. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        logger.error("Failed to capture frame."); break

    frame.flags.writeable = False
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face_mesh = face_mesh.process(rgb_frame)
    frame.flags.writeable = True

    if results_face_mesh.multi_face_landmarks:
        # Reset face missing timer if a face is detected
        FACE_MISSING_START_TIME = None
        FACE_MISSING_ALERT_TRIGGERED = False

        for face_landmarks in results_face_mesh.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            frame_shape = frame.shape
            
            avg_ear = (calculate_ear(landmarks, frame_shape) + calculate_ear(landmarks, frame_shape)) / 2.0
            mouth_opening = get_mouth_opening(landmarks, frame_shape)
            head_pose_angles = get_head_pose(landmarks, frame_shape)

            # --- Eye Closure Check ---
            if avg_ear < EYE_AR_THRESH:
                EYE_COUNTER += 1
                if EYE_COUNTER >= EYE_AR_CONSEC_FRAMES and not EYES_CLOSED_ALERT_TRIGGERED:
                    cv2.putText(frame, "ALERT: EYES CLOSED", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    logger.warning("Prolonged eye closure detected!")
                    EYES_CLOSED_ALERT_TRIGGERED = True
            else:
                EYE_COUNTER, EYES_CLOSED_ALERT_TRIGGERED = 0, False

            # --- Yawn Check ---
            if mouth_opening > MOUTH_OPEN_THRESH:
                MOUTH_COUNTER += 1
                if MOUTH_COUNTER >= MOUTH_OPEN_CONSEC_FRAMES and not YAWN_ALERT_TRIGGERED:
                    cv2.putText(frame, "ALERT: YAWN DETECTED", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    logger.warning("Yawn detected!")
                    YAWN_ALERT_TRIGGERED = True
            else:
                MOUTH_COUNTER, YAWN_ALERT_TRIGGERED = 0, False
            
            if head_pose_angles:
                pitch, yaw, roll = head_pose_angles
                # --- Side-to-Side Tilt (Roll) Check ---
                if abs(roll) > ROLL_THRESH_DEGREES:
                    ROLL_COUNTER += 1
                    if ROLL_COUNTER >= ROLL_CONSEC_FRAMES and not ROLL_ALERT_TRIGGERED:
                        cv2.putText(frame, "ALERT: HEAD TILT (SIDE)", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        logger.warning(f"Side head tilt detected! Angle: {roll:.2f} degrees")
                        ROLL_ALERT_TRIGGERED = True
                else:
                    ROLL_COUNTER, ROLL_ALERT_TRIGGERED = 0, False
                
                # --- Up/Down Tilt (Pitch) Check ---
                if abs(pitch) > PITCH_THRESH_DEGREES:
                    PITCH_COUNTER += 1
                    if PITCH_COUNTER >= PITCH_CONSEC_FRAMES and not PITCH_ALERT_TRIGGERED:
                        cv2.putText(frame, "ALERT: HEAD TILTED UP/DOWN", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                        logger.warning(f"Up/Down head tilt detected! Angle: {pitch:.2f} degrees")
                        PITCH_ALERT_TRIGGERED = True
                else:
                    PITCH_COUNTER, PITCH_ALERT_TRIGGERED = 0, False

            mp_drawing.draw_landmarks(
                image=frame, landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, color=(0, 200, 0))
            )
    else:
        # --- No Face Detected Check ---
        if FACE_MISSING_START_TIME is None:
            FACE_MISSING_START_TIME = time.time()
        elif time.time() - FACE_MISSING_START_TIME > FACE_MISSING_SECONDS:
            cv2.putText(frame, "Please look at the camera", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not FACE_MISSING_ALERT_TRIGGERED:
                logger.warning("Face not detected for a prolonged period.")
                FACE_MISSING_ALERT_TRIGGERED = True

    cv2.imshow('Drowsiness & Attention Monitor', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

# =============================================================================
# 4. CLEANUP
# =============================================================================
logger.info("Shutting down...")
cap.release()
cv2.destroyAllWindows()
logger.info("Cleanup complete. Exiting.")