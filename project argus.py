<<<<<<< HEAD
#!/usr/bin/env python
# -- coding: utf-8 --

=======
>>>>>>> a7aaa2eeedc7bd4064d051ca5ba65f20f6d7fcb2
import cv2
import time
import mediapipe as mp
import math
import logging
import numpy as np
# Note: Removed YOLO imports (ultralytics, torch)

# 1. INITIALIZATION AND SETUP

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

FRAME_W = 640
FRAME_H = 480

<<<<<<< HEAD
# --- Detection Constants ---
EYE_AR_THRESH = 0.21 # Refined
EYE_AR_CONSEC_FRAMES = 30 # Refined
=======

EYE_AR_THRESH = 0.19 

EYE_AR_CONSEC_FRAMES = 20 

>>>>>>> a7aaa2eeedc7bd4064d051ca5ba65f20f6d7fcb2
MOUTH_OPEN_THRESH = 25
MOUTH_OPEN_CONSEC_FRAMES = 25
<<<<<<< HEAD
ROLL_THRESH_DEGREES = 20
ROLL_CONSEC_FRAMES = 20
PITCH_THRESH_DEGREES = 20
PITCH_CONSEC_FRAMES = 20
YAW_THRESH_DEGREES = 25
YAW_CONSEC_FRAMES = 20
=======

ROLL_THRESH_DEGREES = 20
ROLL_CONSEC_FRAMES = 15
PITCH_THRESH_DEGREES = 20
PITCH_CONSEC_FRAMES = 15


YAW_THRESH_DEGREES = 20 
YAW_CONSEC_FRAMES = 15 

>>>>>>> a7aaa2eeedc7bd4064d051ca5ba65f20f6d7fcb2
FACE_MISSING_SECONDS = 3.0
<<<<<<< HEAD
MOUTH_COVERED_THRESH_DISTANCE = 50
MOUTH_COVERED_CONSEC_FRAMES = 15
EYES_COVERED_THRESH_DISTANCE = 50
EYES_COVERED_CONSEC_FRAMES = 15
# Constants for phone gesture detection (using landmarks)
PHONE_GESTURE_DIST_THRESH = 80 # Pixel distance threshold: index finger base to ear
PHONE_GESTURE_CONSEC_FRAMES = 30 # Increased frames
=======
MOUTH_COVERED_THRESH_DISTANCE = 50
MOUTH_COVERED_CONSEC_FRAMES = 15
EYES_COVERED_THRESH_DISTANCE = 50
EYES_COVERED_CONSEC_FRAMES = 15
>>>>>>> a7aaa2eeedc7bd4064d051ca5ba65f20f6d7fcb2

<<<<<<< HEAD
# Thresholds for cumulative break alert
YAWN_COUNT_THRESH = 5
DROWSINESS_COUNT_THRESH = 10

# Duration for alerts to stay on screen
ALERT_DURATION_SECONDS = 5.0

# Landmark indices
LEFT_EYE_INDICES = [33, 133, 159, 145, 158, 153]
RIGHT_EYE_INDICES = [263, 362, 386, 374, 385, 380]
LEFT_EAR_INDEX = 361
RIGHT_EAR_INDEX = 132

# --- State counters and alert flags ---
EYE_COUNTER, MOUTH_COUNTER = 0, 0
POSTURE_COUNTER = 0
YAW_COUNTER = 0
MOUTH_COVERED_COUNTER = 0
EYES_COVERED_COUNTER = 0
PHONE_GESTURE_COUNTER = 0
# Cumulative counters for break alert
CUMULATIVE_YAWN_COUNT = 0
CUMULATIVE_DROWSINESS_COUNT = 0

=======
PHONE_GESTURE_DIST_THRESH = 80 
PHONE_GESTURE_CONSEC_FRAMES = 30 

YAWN_COUNT_THRESH = 5
DROWSINESS_COUNT_THRESH = 10

ALERT_DURATION_SECONDS = 5.0

LEFT_EYE_INDICES = [33, 133, 159, 145, 158, 153]
RIGHT_EYE_INDICES = [263, 362, 386, 374, 385, 380]
LEFT_EAR_INDEX = 361
RIGHT_EAR_INDEX = 132

EYE_COUNTER, MOUTH_COUNTER = 0, 0
POSTURE_COUNTER = 0
YAW_COUNTER = 0
MOUTH_COVERED_COUNTER = 0
EYES_COVERED_COUNTER = 0
PHONE_GESTURE_COUNTER = 0

CUMULATIVE_YAWN_COUNT = 0
CUMULATIVE_DROWSINESS_COUNT = 0

>>>>>>> a7aaa2eeedc7bd4064d051ca5ba65f20f6d7fcb2
EYES_CLOSED_ALERT_TRIGGERED, YAWN_ALERT_TRIGGERED = False, False
POSTURE_ALERT_TRIGGERED = False
YAW_ALERT_TRIGGERED = False
FACE_MISSING_START_TIME, FACE_MISSING_ALERT_TRIGGERED = None, False
MOUTH_COVERED_ALERT_TRIGGERED = False
EYES_COVERED_ALERT_TRIGGERED = False
HANDS_OFF_WHEEL_ALERT_TRIGGERED = False
TAKE_BREAK_ALERT_TRIGGERED = False
PHONE_GESTURE_ALERT_TRIGGERED = False

<<<<<<< HEAD

# Timestamps for when to STOP displaying alerts
EYES_CLOSED_DISPLAY_UNTIL = 0.0
YAWN_DISPLAY_UNTIL = 0.0
POSTURE_DISPLAY_UNTIL = 0.0
YAW_DISPLAY_UNTIL = 0.0
MOUTH_COVERED_DISPLAY_UNTIL = 0.0
EYES_COVERED_DISPLAY_UNTIL = 0.0
HANDS_OFF_WHEEL_DISPLAY_UNTIL = 0.0
FACE_MISSING_DISPLAY_UNTIL = 0.0
TAKE_BREAK_DISPLAY_UNTIL = 0.0
PHONE_GESTURE_DISPLAY_UNTIL = 0.0


# --- MediaPipe Models Setup ---
=======


EYES_CLOSED_DISPLAY_UNTIL = 0.0
YAWN_DISPLAY_UNTIL = 0.0
POSTURE_DISPLAY_UNTIL = 0.0
YAW_DISPLAY_UNTIL = 0.0
MOUTH_COVERED_DISPLAY_UNTIL = 0.0
EYES_COVERED_DISPLAY_UNTIL = 0.0
HANDS_OFF_WHEEL_DISPLAY_UNTIL = 0.0
FACE_MISSING_DISPLAY_UNTIL = 0.0
TAKE_BREAK_DISPLAY_UNTIL = 0.0
PHONE_GESTURE_DISPLAY_UNTIL = 0.0


>>>>>>> a7aaa2eeedc7bd4064d051ca5ba65f20f6d7fcb2
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7
)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error("Could not open video stream. Make sure a webcam is connected.")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
time.sleep(2)

# 2. HELPER FUNCTIONS

<<<<<<< HEAD
# ... (calculate_ear, get_mouth_opening, get_head_pose functions remain the same) ...
def calculate_ear(eye_landmarks, frame_shape, indices):
    """Calculates Eye Aspect Ratio (EAR) given specific landmark indices."""
    try:
        max_idx = max(indices)
        if max_idx >= len(eye_landmarks): return 0.3
        p1 = (int(eye_landmarks[indices[0]].x * frame_shape[1]), int(eye_landmarks[indices[0]].y * frame_shape[0]))
        p4 = (int(eye_landmarks[indices[1]].x * frame_shape[1]), int(eye_landmarks[indices[1]].y * frame_shape[0]))
        p2 = (int(eye_landmarks[indices[2]].x * frame_shape[1]), int(eye_landmarks[indices[2]].y * frame_shape[0]))
        p6 = (int(eye_landmarks[indices[3]].x * frame_shape[1]), int(eye_landmarks[indices[3]].y * frame_shape[0]))
        p3 = (int(eye_landmarks[indices[4]].x * frame_shape[1]), int(eye_landmarks[indices[4]].y * frame_shape[0]))
        p5 = (int(eye_landmarks[indices[5]].x * frame_shape[1]), int(eye_landmarks[indices[5]].y * frame_shape[0]))
        ver_dist1 = math.hypot(p2[0] - p6[0], p2[1] - p6[1])
        ver_dist2 = math.hypot(p3[0] - p5[0], p3[1] - p5[1])
        hor_dist = math.hypot(p1[0] - p4[0], p1[1] - p4[1])
        if hor_dist < 1e-6: return 0.3
        return (ver_dist1 + ver_dist2) / (2.0 * hor_dist)
    except Exception as e:
        logger.error(f"Unexpected error in calculate_ear: {e}")
        return 0.3
=======
>>>>>>> a7aaa2eeedc7bd4064d051ca5ba65f20f6d7fcb2


def calculate_ear(eye_landmarks, frame_shape, indices):
    """Calculates Eye Aspect Ratio (EAR) given specific landmark indices."""
    try:
        max_idx = max(indices)
        if max_idx >= len(eye_landmarks): return 0.3
        p1 = (int(eye_landmarks[indices[0]].x * frame_shape[1]), int(eye_landmarks[indices[0]].y * frame_shape[0]))
        p4 = (int(eye_landmarks[indices[1]].x * frame_shape[1]), int(eye_landmarks[indices[1]].y * frame_shape[0]))
        p2 = (int(eye_landmarks[indices[2]].x * frame_shape[1]), int(eye_landmarks[indices[2]].y * frame_shape[0]))
        p6 = (int(eye_landmarks[indices[3]].x * frame_shape[1]), int(eye_landmarks[indices[3]].y * frame_shape[0]))
        p3 = (int(eye_landmarks[indices[4]].x * frame_shape[1]), int(eye_landmarks[indices[4]].y * frame_shape[0]))
        p5 = (int(eye_landmarks[indices[5]].x * frame_shape[1]), int(eye_landmarks[indices[5]].y * frame_shape[0]))
        ver_dist1 = math.hypot(p2[0] - p6[0], p2[1] - p6[1])
        ver_dist2 = math.hypot(p3[0] - p5[0], p3[1] - p5[1])
        hor_dist = math.hypot(p1[0] - p4[0], p1[1] - p4[1])
        if hor_dist < 1e-6: return 0.3
        return (ver_dist1 + ver_dist2) / (2.0 * hor_dist)
    except Exception as e:
        logger.error(f"Unexpected error in calculate_ear: {e}")
        return 0.3

def get_mouth_opening(mouth_landmarks, frame_shape):
    try:
        if 13 >= len(mouth_landmarks) or 14 >= len(mouth_landmarks): return 0
        upper_lip = (int(mouth_landmarks[13].x * frame_shape[1]), int(mouth_landmarks[13].y * frame_shape[0]))
        lower_lip = (int(mouth_landmarks[14].x * frame_shape[1]), int(mouth_landmarks[14].y * frame_shape[0]))
        return math.hypot(upper_lip[0] - lower_lip[0], upper_lip[1] - lower_lip[1])
    except Exception as e:
        logger.error(f"Unexpected error in get_mouth_opening: {e}")
        return 0

def get_head_pose(landmarks, frame_shape):
    try:
        h, w, _ = frame_shape
        face_model_3d = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)], dtype=np.float64)
        pnp_landmark_ids = [1, 152, 263, 33, 291, 61]
        max_landmark_idx = max(pnp_landmark_ids)
        if max_landmark_idx >= len(landmarks): return None
        image_points_2d = np.array([(landmarks[id].x * w, landmarks[id].y * h) for id in pnp_landmark_ids], dtype=np.float64)
        if np.any(np.isnan(image_points_2d)) or np.any(np.isinf(image_points_2d)): return None
        focal_length = w; center = (w/2, h/2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype = "double")
        dist_coeffs = np.zeros((4,1))
        success, rotation_vector, translation_vector = cv2.solvePnP(face_model_3d, image_points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success: return None
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)
        if euler_angles is None or len(euler_angles) == 0: return None
        pitch = euler_angles[0, 0]; yaw = euler_angles[1, 0]; roll = euler_angles[2, 0]
        return pitch, yaw, roll
    except Exception as e:
        logger.error(f"Unexpected error in get_head_pose: {e}", exc_info=True)
        return None

<<<<<<< HEAD
# =============================================================================
=======

>>>>>>> a7aaa2eeedc7bd4064d051ca5ba65f20f6d7fcb2
# 3. MAIN PROCESSING LOOP

logger.info("Starting video processing loop. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        logger.error("Failed to capture frame."); break

<<<<<<< HEAD
    frame = cv2.flip(frame, 1) # Flip horizontally
=======
    frame = cv2.flip(frame, 1) 
>>>>>>> a7aaa2eeedc7bd4064d051ca5ba65f20f6d7fcb2

<<<<<<< HEAD
    # --- MediaPipe Processing ---
    frame_rgb_mp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb_mp.flags.writeable = False
    results_face_mesh = face_mesh.process(frame_rgb_mp)
    results_hands = hands.process(frame_rgb_mp)
    frame.flags.writeable = True # Make original frame writeable for drawing

    # --- Process MediaPipe Face Results ---
=======

    frame_rgb_mp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb_mp.flags.writeable = False
    results_face_mesh = face_mesh.process(frame_rgb_mp)
    results_hands = hands.process(frame_rgb_mp)
    frame.flags.writeable = True 

   
>>>>>>> a7aaa2eeedc7bd4064d051ca5ba65f20f6d7fcb2
    if results_face_mesh.multi_face_landmarks:
        FACE_MISSING_START_TIME = None
        FACE_MISSING_ALERT_TRIGGERED = False

<<<<<<< HEAD
        face_landmarks_list = results_face_mesh.multi_face_landmarks[0].landmark
        frame_shape = frame.shape
=======
        face_landmarks_list = results_face_mesh.multi_face_landmarks[0].landmark
        frame_shape = frame.shape

        left_ear_val = calculate_ear(face_landmarks_list, frame_shape, LEFT_EYE_INDICES)
        right_ear_val = calculate_ear(face_landmarks_list, frame_shape, RIGHT_EYE_INDICES)
        avg_ear = ((left_ear_val + right_ear_val) / 2.0) if left_ear_val is not None and right_ear_val is not None else 0.3
        
       

        mouth_opening = get_mouth_opening(face_landmarks_list, frame_shape)
        head_pose_angles = get_head_pose(face_landmarks_list, frame_shape)

        
        if avg_ear < EYE_AR_THRESH:
            EYE_COUNTER += 1
            if EYE_COUNTER >= EYE_AR_CONSEC_FRAMES and not EYES_CLOSED_ALERT_TRIGGERED:
                EYES_CLOSED_DISPLAY_UNTIL = time.time() + ALERT_DURATION_SECONDS
                logger.warning("Prolonged eye closure detected!")
                EYES_CLOSED_ALERT_TRIGGERED = True
                CUMULATIVE_DROWSINESS_COUNT += 1
        else:
            EYE_COUNTER, EYES_CLOSED_ALERT_TRIGGERED = 0, False

    
        if mouth_opening > MOUTH_OPEN_THRESH:
            MOUTH_COUNTER += 1
            if MOUTH_COUNTER >= MOUTH_OPEN_CONSEC_FRAMES and not YAWN_ALERT_TRIGGERED:
                YAWN_DISPLAY_UNTIL = time.time() + ALERT_DURATION_SECONDS
                logger.warning("Yawn detected!")
                YAWN_ALERT_TRIGGERED = True
                CUMULATIVE_YAWN_COUNT += 1
        else:
            MOUTH_COUNTER, YAWN_ALERT_TRIGGERED = 0, False

        
        if (CUMULATIVE_DROWSINESS_COUNT > DROWSINESS_COUNT_THRESH or CUMULATIVE_YAWN_COUNT > YAWN_COUNT_THRESH) and not TAKE_BREAK_ALERT_TRIGGERED:
            TAKE_BREAK_DISPLAY_UNTIL = time.time() + ALERT_DURATION_SECONDS
            logger.warning("Critical fatigue detected! Advising driver to take a break.")
            TAKE_BREAK_ALERT_TRIGGERED = True

        
        if head_pose_angles:
            pitch, yaw, roll = head_pose_angles
            
       
>>>>>>> a7aaa2eeedc7bd4064d051ca5ba65f20f6d7fcb2

<<<<<<< HEAD
        left_ear_val = calculate_ear(face_landmarks_list, frame_shape, LEFT_EYE_INDICES)
        right_ear_val = calculate_ear(face_landmarks_list, frame_shape, RIGHT_EYE_INDICES)
        avg_ear = ((left_ear_val + right_ear_val) / 2.0) if left_ear_val is not None and right_ear_val is not None else 0.3

        mouth_opening = get_mouth_opening(face_landmarks_list, frame_shape)
        head_pose_angles = get_head_pose(face_landmarks_list, frame_shape)

        # Eye Closure Check
        if avg_ear < EYE_AR_THRESH:
            EYE_COUNTER += 1
            if EYE_COUNTER >= EYE_AR_CONSEC_FRAMES and not EYES_CLOSED_ALERT_TRIGGERED:
                EYES_CLOSED_DISPLAY_UNTIL = time.time() + ALERT_DURATION_SECONDS
                logger.warning("Prolonged eye closure detected!")
                EYES_CLOSED_ALERT_TRIGGERED = True
                CUMULATIVE_DROWSINESS_COUNT += 1
        else:
            EYE_COUNTER, EYES_CLOSED_ALERT_TRIGGERED = 0, False

        # Yawn Check
        if mouth_opening > MOUTH_OPEN_THRESH:
            MOUTH_COUNTER += 1
            if MOUTH_COUNTER >= MOUTH_OPEN_CONSEC_FRAMES and not YAWN_ALERT_TRIGGERED:
                YAWN_DISPLAY_UNTIL = time.time() + ALERT_DURATION_SECONDS
                logger.warning("Yawn detected!")
                YAWN_ALERT_TRIGGERED = True
                CUMULATIVE_YAWN_COUNT += 1
        else:
            MOUTH_COUNTER, YAWN_ALERT_TRIGGERED = 0, False

        # "Take a Break" Check
        if (CUMULATIVE_DROWSINESS_COUNT > DROWSINESS_COUNT_THRESH or CUMULATIVE_YAWN_COUNT > YAWN_COUNT_THRESH) and not TAKE_BREAK_ALERT_TRIGGERED:
            TAKE_BREAK_DISPLAY_UNTIL = time.time() + ALERT_DURATION_SECONDS
            logger.warning("Critical fatigue detected! Advising driver to take a break.")
            TAKE_BREAK_ALERT_TRIGGERED = True

        # Head Pose Checks
        if head_pose_angles:
            pitch, yaw, roll = head_pose_angles
            if abs(yaw) > YAW_THRESH_DEGREES:
                YAW_COUNTER += 1
                if YAW_COUNTER >= YAW_CONSEC_FRAMES and not YAW_ALERT_TRIGGERED:
                    YAW_DISPLAY_UNTIL = time.time() + ALERT_DURATION_SECONDS
                    logger.warning(f"Head turned sideways! Yaw: {yaw:.2f}")
                    YAW_ALERT_TRIGGERED = True
            else:
                YAW_COUNTER, YAW_ALERT_TRIGGERED = 0, False
            if abs(yaw) < YAW_THRESH_DEGREES:
                if abs(roll) > ROLL_THRESH_DEGREES or abs(pitch) > PITCH_THRESH_DEGREES:
                    POSTURE_COUNTER += 1
                    if POSTURE_COUNTER >= ROLL_CONSEC_FRAMES and not POSTURE_ALERT_TRIGGERED:
                        POSTURE_DISPLAY_UNTIL = time.time() + ALERT_DURATION_SECONDS
                        logger.warning(f"Head tilt! Roll: {roll:.2f}, Pitch: {pitch:.2f}")
                        POSTURE_ALERT_TRIGGERED = True
=======
            if abs(yaw) > YAW_THRESH_DEGREES:
                YAW_COUNTER += 1
                if YAW_COUNTER >= YAW_CONSEC_FRAMES and not YAW_ALERT_TRIGGERED:
                    YAW_DISPLAY_UNTIL = time.time() + ALERT_DURATION_SECONDS
                    logger.warning(f"Head turned sideways! Yaw: {yaw:.2f}")
                    YAW_ALERT_TRIGGERED = True
            else:
                YAW_COUNTER, YAW_ALERT_TRIGGERED = 0, False
            
            if abs(yaw) < YAW_THRESH_DEGREES: # Only check posture if not looking sideways
                if abs(roll) > ROLL_THRESH_DEGREES or abs(pitch) > PITCH_THRESH_DEGREES:
                    POSTURE_COUNTER += 1
                    if POSTURE_COUNTER >= ROLL_CONSEC_FRAMES and not POSTURE_ALERT_TRIGGERED:
                        POSTURE_DISPLAY_UNTIL = time.time() + ALERT_DURATION_SECONDS
                        logger.warning(f"Head tilt! Roll: {roll:.2f}, Pitch: {pitch:.2f}")
                        POSTURE_ALERT_TRIGGERED = True
>>>>>>> a7aaa2eeedc7bd4064d051ca5ba65f20f6d7fcb2
                else:
<<<<<<< HEAD
                    POSTURE_COUNTER, POSTURE_ALERT_TRIGGERED = 0, False
            else:
                POSTURE_COUNTER, POSTURE_ALERT_TRIGGERED = 0, False
        else:
             YAW_COUNTER, YAW_ALERT_TRIGGERED = 0, False
             POSTURE_COUNTER, POSTURE_ALERT_TRIGGERED = 0, False
=======
                    POSTURE_COUNTER, POSTURE_ALERT_TRIGGERED = 0, False
            else:
                POSTURE_COUNTER, POSTURE_ALERT_TRIGGERED = 0, False
        else:
             YAW_COUNTER, YAW_ALERT_TRIGGERED = 0, False
             POSTURE_COUNTER, POSTURE_ALERT_TRIGGERED = 0, False


       
        is_mouth_covered = False
        is_eyes_covered = False
        is_phone_gesture = False

        if results_hands.multi_hand_landmarks:
            h, w, _ = frame_shape
            try:
             
                nose_tip_y = int(face_landmarks_list[1].y * h)
                chin_y = int(face_landmarks_list[152].y * h) 
                eyebrow_y = int(((face_landmarks_list[105].y + face_landmarks_list[334].y) / 2) * h)
                
                mouth_center_x = int(((face_landmarks_list[13].x + face_landmarks_list[14].x) / 2) * w)
                mouth_center_y = int(((face_landmarks_list[13].y + face_landmarks_list[14].y) / 2) * h)
                left_eye_x = int(face_landmarks_list[159].x * w); left_eye_y = int(face_landmarks_list[159].y * h)
                right_eye_x = int(face_landmarks_list[386].x * w); right_eye_y = int(face_landmarks_list[386].y * h)
                left_ear_x = int(face_landmarks_list[LEFT_EAR_INDEX].x * w); left_ear_y = int(face_landmarks_list[LEFT_EAR_INDEX].y * h)
                right_ear_x = int(face_landmarks_list[RIGHT_EAR_INDEX].x * w); right_ear_y = int(face_landmarks_list[RIGHT_EAR_INDEX].y * h)
                
                
                HAND_COVERING_INDICES = [
                    mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
                    mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_MCP
                ]
>>>>>>> a7aaa2eeedc7bd4064d051ca5ba65f20f6d7fcb2

<<<<<<< HEAD

        # --- Hand Proximity Checks (Only if face detected) ---
        is_mouth_covered = False
        is_eyes_covered = False
        is_phone_gesture = False

        if results_hands.multi_hand_landmarks:
            h, w, _ = frame_shape
            try:
                nose_tip_y = int(face_landmarks_list[1].y * h)
                mouth_center_x = int(((face_landmarks_list[13].x + face_landmarks_list[14].x) / 2) * w)
                mouth_center_y = int(((face_landmarks_list[13].y + face_landmarks_list[14].y) / 2) * h)
                left_eye_x = int(face_landmarks_list[159].x * w); left_eye_y = int(face_landmarks_list[159].y * h)
                right_eye_x = int(face_landmarks_list[386].x * w); right_eye_y = int(face_landmarks_list[386].y * h)
                left_ear_x = int(face_landmarks_list[LEFT_EAR_INDEX].x * w); left_ear_y = int(face_landmarks_list[LEFT_EAR_INDEX].y * h)
                right_ear_x = int(face_landmarks_list[RIGHT_EAR_INDEX].x * w); right_ear_y = int(face_landmarks_list[RIGHT_EAR_INDEX].y * h)

                for hand_landmarks in results_hands.multi_hand_landmarks:
                    # Phone Gesture Check (using index finger base)
                    try:
                        idx_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                        idx_mcp_x = int(idx_mcp.x * w); idx_mcp_y = int(idx_mcp.y * h)
                        dist_idx_mcp_left_ear = math.hypot(idx_mcp_x - left_ear_x, idx_mcp_y - left_ear_y)
                        dist_idx_mcp_right_ear = math.hypot(idx_mcp_x - right_ear_x, idx_mcp_y - right_ear_y)
                        if (dist_idx_mcp_left_ear < PHONE_GESTURE_DIST_THRESH or dist_idx_mcp_right_ear < PHONE_GESTURE_DIST_THRESH) and idx_mcp_y < mouth_center_y :
                             is_phone_gesture = True
                    except IndexError:
                         logger.warning("Hand landmark index out of range for phone check.")

                    # Covered Checks Loop
                    for lm_idx, lm in enumerate(hand_landmarks.landmark):
                        hand_lm_x = int(lm.x * w)
                        hand_lm_y = int(lm.y * h)
                        # Check Mouth Covered
                        mouth_dist = math.hypot(hand_lm_x - mouth_center_x, hand_lm_y - mouth_center_y)
                        if mouth_dist < MOUTH_COVERED_THRESH_DISTANCE and hand_lm_y >= nose_tip_y:
                            is_mouth_covered = True
                        # Check Eyes Covered
                        left_eye_dist = math.hypot(hand_lm_x - left_eye_x, hand_lm_y - left_eye_y)
                        right_eye_dist = math.hypot(hand_lm_x - right_eye_x, hand_lm_y - right_eye_y)
                        if (left_eye_dist < EYES_COVERED_THRESH_DISTANCE or right_eye_dist < EYES_COVERED_THRESH_DISTANCE) and hand_lm_y < nose_tip_y:
                            is_eyes_covered = True
                        if is_eyes_covered: break
                    if is_eyes_covered: break

            except IndexError:
                 logger.error("Face landmark index out of range during hand proximity checks.")
            except Exception as e:
                 logger.error(f"Error during hand proximity checks: {e}")


        # Hierarchy for Hand Alerts
        if is_eyes_covered:
            is_mouth_covered = False
            is_phone_gesture = False
        elif is_mouth_covered:
            is_phone_gesture = False

        # Alert Logic for Hand Gestures/Proximity
        # Eyes Covered
        if is_eyes_covered:
            EYES_COVERED_COUNTER += 1
            if EYES_COVERED_COUNTER >= EYES_COVERED_CONSEC_FRAMES and not EYES_COVERED_ALERT_TRIGGERED:
                EYES_COVERED_DISPLAY_UNTIL = time.time() + ALERT_DURATION_SECONDS
                logger.warning("Eyes covered detected!")
                EYES_COVERED_ALERT_TRIGGERED = True
        else:
            EYES_COVERED_COUNTER, EYES_COVERED_ALERT_TRIGGERED = 0, False
        # Mouth Covered
        if is_mouth_covered:
            MOUTH_COVERED_COUNTER += 1
            if MOUTH_COVERED_COUNTER >= MOUTH_COVERED_CONSEC_FRAMES and not MOUTH_COVERED_ALERT_TRIGGERED:
                MOUTH_COVERED_DISPLAY_UNTIL = time.time() + ALERT_DURATION_SECONDS
                logger.warning("Mouth covered detected!")
                MOUTH_COVERED_ALERT_TRIGGERED = True
        else:
            MOUTH_COVERED_COUNTER, MOUTH_COVERED_ALERT_TRIGGERED = 0, False
        # Phone Gesture
        if is_phone_gesture:
            PHONE_GESTURE_COUNTER += 1
            if PHONE_GESTURE_COUNTER >= PHONE_GESTURE_CONSEC_FRAMES and not PHONE_GESTURE_ALERT_TRIGGERED:
                PHONE_GESTURE_DISPLAY_UNTIL = time.time() + ALERT_DURATION_SECONDS
                logger.warning("Phone gesture detected!")
                PHONE_GESTURE_ALERT_TRIGGERED = True
        else:
            # Only reset counter/flag if the alert is not currently being displayed
            if time.time() > PHONE_GESTURE_DISPLAY_UNTIL:
                PHONE_GESTURE_COUNTER, PHONE_GESTURE_ALERT_TRIGGERED = 0, False


        # Draw face landmarks
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=results_face_mesh.multi_face_landmarks[0],
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
            connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, color=(0, 200, 0))
        )

    else: # No face detected
=======
                for hand_landmarks in results_hands.multi_hand_landmarks:
                   
                    try:
                        idx_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                        idx_mcp_x = int(idx_mcp.x * w); idx_mcp_y = int(idx_mcp.y * h)
                        dist_idx_mcp_left_ear = math.hypot(idx_mcp_x - left_ear_x, idx_mcp_y - left_ear_y)
                        dist_idx_mcp_right_ear = math.hypot(idx_mcp_x - right_ear_x, idx_mcp_y - right_ear_y)
                        if (dist_idx_mcp_left_ear < PHONE_GESTURE_DIST_THRESH or dist_idx_mcp_right_ear < PHONE_GESTURE_DIST_THRESH) and idx_mcp_y < mouth_center_y :
                            is_phone_gesture = True
                    except IndexError:
                        logger.warning("Hand landmark index out of range for phone check.")

                   
                    for lm_idx in HAND_COVERING_INDICES:
                        try:
                            lm = hand_landmarks.landmark[lm_idx]
                            hand_lm_x = int(lm.x * w)
                            hand_lm_y = int(lm.y * h)
                            
                        
                            mouth_dist = math.hypot(hand_lm_x - mouth_center_x, hand_lm_y - mouth_center_y)
                           
                            if mouth_dist < MOUTH_COVERED_THRESH_DISTANCE and (nose_tip_y < hand_lm_y < chin_y):
                                is_mouth_covered = True

                            left_eye_dist = math.hypot(hand_lm_x - left_eye_x, hand_lm_y - left_eye_y)
                            right_eye_dist = math.hypot(hand_lm_x - right_eye_x, hand_lm_y - right_eye_y)
                           
                            if (left_eye_dist < EYES_COVERED_THRESH_DISTANCE or right_eye_dist < EYES_COVERED_THRESH_DISTANCE) and (eyebrow_y < hand_lm_y < nose_tip_y):
                                is_eyes_covered = True
                                break 

                        except IndexError:
                            logger.warning(f"Hand landmark index {lm_idx} out of range.")
                    
                    if is_eyes_covered: 
                        break 

            except IndexError:
                logger.error("Face landmark index out of range during hand proximity checks.")
            except Exception as e:
                logger.error(f"Error during hand proximity checks: {e}")


       
        if is_eyes_covered:
            is_mouth_covered = False
            is_phone_gesture = False
        elif is_mouth_covered:
            is_phone_gesture = False

        
        if is_eyes_covered:
            EYES_COVERED_COUNTER += 1
            if EYES_COVERED_COUNTER >= EYES_COVERED_CONSEC_FRAMES and not EYES_COVERED_ALERT_TRIGGERED:
                EYES_COVERED_DISPLAY_UNTIL = time.time() + ALERT_DURATION_SECONDS
                logger.warning("Eyes covered detected!")
                EYES_COVERED_ALERT_TRIGGERED = True
        else:
            EYES_COVERED_COUNTER, EYES_COVERED_ALERT_TRIGGERED = 0, False
       
        if is_mouth_covered:
            MOUTH_COVERED_COUNTER += 1
            if MOUTH_COUNTER >= MOUTH_COVERED_CONSEC_FRAMES and not MOUTH_COVERED_ALERT_TRIGGERED:
                MOUTH_COVERED_DISPLAY_UNTIL = time.time() + ALERT_DURATION_SECONDS
                logger.warning("Mouth covered detected!")
                MOUTH_COVERED_ALERT_TRIGGERED = True
        else:
            MOUTH_COVERED_COUNTER, MOUTH_COVERED_ALERT_TRIGGERED = 0, False

        if is_phone_gesture:
            PHONE_GESTURE_COUNTER += 1
            if PHONE_GESTURE_COUNTER >= PHONE_GESTURE_CONSEC_FRAMES and not PHONE_GESTURE_ALERT_TRIGGERED:
                PHONE_GESTURE_DISPLAY_UNTIL = time.time() + ALERT_DURATION_SECONDS
                logger.warning("Phone gesture detected!")
                PHONE_GESTURE_ALERT_TRIGGERED = True
        else:
           
            if time.time() > PHONE_GESTURE_DISPLAY_UNTIL:
                PHONE_GESTURE_COUNTER, PHONE_GESTURE_ALERT_TRIGGERED = 0, False



        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=results_face_mesh.multi_face_landmarks[0],
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
            connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, color=(0, 200, 0))
        )

    else: 
>>>>>>> a7aaa2eeedc7bd4064d051ca5ba65f20f6d7fcb2
        if FACE_MISSING_START_TIME is None:
            FACE_MISSING_START_TIME = time.time()
        elif time.time() - FACE_MISSING_START_TIME > FACE_MISSING_SECONDS:
            if not FACE_MISSING_ALERT_TRIGGERED:
<<<<<<< HEAD
                FACE_MISSING_DISPLAY_UNTIL = time.time() + ALERT_DURATION_SECONDS
                logger.warning("Face not detected for a prolonged period.")
=======
                FACE_MISSING_DISPLAY_UNTIL = time.time() + ALERT_DURATION_SECONDS
                logger.warning("Face not-detected for a prolonged period.")
>>>>>>> a7aaa2eeedc7bd4064d051ca5ba65f20f6d7fcb2
                FACE_MISSING_ALERT_TRIGGERED = True
        YAW_COUNTER, YAW_ALERT_TRIGGERED = 0, False
        POSTURE_COUNTER, POSTURE_ALERT_TRIGGERED = 0, False
        PHONE_GESTURE_COUNTER, PHONE_GESTURE_ALERT_TRIGGERED = 0, False

<<<<<<< HEAD

    # --- Hands Off Wheel Logic ---
    hands_detected = results_hands.multi_hand_landmarks is not None
    current_time_for_hand_check = time.time()
    # Check if any specific hand alert is currently active or displaying
    ## UPDATE: Simplified check using just the _TRIGGERED flags for active alerts
    specific_hand_alert_active = (PHONE_GESTURE_ALERT_TRIGGERED or
                                  EYES_COVERED_ALERT_TRIGGERED or
                                  MOUTH_COVERED_ALERT_TRIGGERED)

    if hands_detected:
        if not specific_hand_alert_active: # No specific alerts active? Check general hands-off.
            if not HANDS_OFF_WHEEL_ALERT_TRIGGERED:
                HANDS_OFF_WHEEL_DISPLAY_UNTIL = time.time() + ALERT_DURATION_SECONDS
                logger.warning("Hand(s) detected off wheel.")
                HANDS_OFF_WHEEL_ALERT_TRIGGERED = True
        else: # Specific alert IS active, ensure general "hands off" is OFF.
            HANDS_OFF_WHEEL_ALERT_TRIGGERED = False
            HANDS_OFF_WHEEL_DISPLAY_UNTIL = 0.0 # Make sure it disappears immediately


        # Draw hand landmarks
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        HANDS_OFF_WHEEL_ALERT_TRIGGERED = False # No hands, so reset


    # =============================================================================
    # 4. DRAW TIMED ALERTS
    # =============================================================================
    current_time = time.time() # Re-get current time for drawing checks

    y_pos = 30 # Starting Y position for alerts
    spacing = 30 # Vertical space between alerts

    # Break alert
    if current_time < TAKE_BREAK_DISPLAY_UNTIL:
        cv2.putText(frame, "Please take a break", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_pos += spacing

    # Drowsiness alert
    if current_time < EYES_CLOSED_DISPLAY_UNTIL:
        cv2.putText(frame, "severe drowsiness", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_pos += spacing

    # Yawn alert
    if current_time < YAWN_DISPLAY_UNTIL:
        cv2.putText(frame, "yawn detected", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += spacing

    # Posture alert
    if current_time < POSTURE_DISPLAY_UNTIL:
        cv2.putText(frame, "adjust posture", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        y_pos += spacing

    # Distraction alert
    if current_time < YAW_DISPLAY_UNTIL:
        cv2.putText(frame, "avoid distractions", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        y_pos += spacing

    # Mouth covered alert
    if current_time < MOUTH_COVERED_DISPLAY_UNTIL:
        cv2.putText(frame, "mouth covered", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        y_pos += spacing

    # Eyes covered alert
    if current_time < EYES_COVERED_DISPLAY_UNTIL:
        cv2.putText(frame, "eyes covered", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        y_pos += spacing

    # Phone gesture alert
    if current_time < PHONE_GESTURE_DISPLAY_UNTIL:
        cv2.putText(frame, "Keep the phone down", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        y_pos += spacing

    # Hands off wheel alert - Only show if no other specific hand/phone alert is currently active
    ## UPDATE: Simplified check using just the _TRIGGERED flags
    specific_hand_alert_active_now = (PHONE_GESTURE_ALERT_TRIGGERED or
                                      EYES_COVERED_ALERT_TRIGGERED or
                                      MOUTH_COVERED_ALERT_TRIGGERED)
    if current_time < HANDS_OFF_WHEEL_DISPLAY_UNTIL and not specific_hand_alert_active_now:
        cv2.putText(frame, "Please keep your hands on the driving wheel", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += spacing


    # Face missing alert (adjust position if needed, check for overlap)
    if current_time < FACE_MISSING_DISPLAY_UNTIL:
        # Place it at a fixed position, potentially overlapping if many alerts are active
        cv2.putText(frame, "Please look at the camera", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


=======


    hands_detected = results_hands.multi_hand_landmarks is not None
    

    specific_hand_alert_active = (PHONE_GESTURE_ALERT_TRIGGERED or
                                  EYES_COVERED_ALERT_TRIGGERED or
                                  MOUTH_COVERED_ALERT_TRIGGERED)
    

    head_distraction_active = (YAW_ALERT_TRIGGERED or POSTURE_ALERT_TRIGGERED)

    if hands_detected:
        
        if not specific_hand_alert_active and not head_distraction_active: 
            if not HANDS_OFF_WHEEL_ALERT_TRIGGERED:
                HANDS_OFF_WHEEL_DISPLAY_UNTIL = time.time() + ALERT_DURATION_SECONDS
                logger.warning("Hand(s) detected off wheel.")
                
                HANDS_OFF_WHEEL_ALERT_TRIGGERED = True
        else: 
            HANDS_OFF_WHEEL_ALERT_TRIGGERED = False
            HANDS_OFF_WHEEL_DISPLAY_UNTIL = 0.0 


        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        HANDS_OFF_WHEEL_ALERT_TRIGGERED = False 


    
   
    current_time = time.time() 

    y_pos = 30 
    spacing = 30 


    if current_time < TAKE_BREAK_DISPLAY_UNTIL:
        cv2.putText(frame, "Please take a break", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_pos += spacing


    if current_time < EYES_CLOSED_DISPLAY_UNTIL:
        cv2.putText(frame, "severe drowsiness", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_pos += spacing


    if current_time < YAWN_DISPLAY_UNTIL:
        cv2.putText(frame, "yawn detected", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += spacing


    if current_time < POSTURE_DISPLAY_UNTIL:
        cv2.putText(frame, "adjust posture", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        y_pos += spacing


    if current_time < YAW_DISPLAY_UNTIL:
        cv2.putText(frame, "avoid distractions", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        y_pos += spacing

    if current_time < MOUTH_COVERED_DISPLAY_UNTIL:
        cv2.putText(frame, "mouth covered", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        y_pos += spacing

    if current_time < EYES_COVERED_DISPLAY_UNTIL:
        cv2.putText(frame, "eyes covered", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        y_pos += spacing


    if current_time < PHONE_GESTURE_DISPLAY_UNTIL:
        cv2.putText(frame, "Keep the phone down", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        y_pos += spacing

    if current_time < HANDS_OFF_WHEEL_DISPLAY_UNTIL and not specific_hand_alert_active:
        cv2.putText(frame, "Please keep your hands on the driving wheel", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += spacing


    
    if current_time < FACE_MISSING_DISPLAY_UNTIL:
        cv2.putText(frame, "Please look at the camera", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


>>>>>>> a7aaa2eeedc7bd4064d051ca5ba65f20f6d7fcb2
    cv2.imshow('Drowsiness & Attention Monitor', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

<<<<<<< HEAD
# =============================================================================
# 5. CLEANUP
# =============================================================================
=======
#cleanup

>>>>>>> a7aaa2eeedc7bd4064d051ca5ba65f20f6d7fcb2
logger.info("Shutting down...")
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
hands.close()
logger.info("Cleanup complete. Exiting.")
