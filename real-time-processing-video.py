import cv2
import subprocess
import random
import mediapipe as mp
import numpy as np

PLAY_AUDIO = True
# POINT_COLOR = random.randint(0, 255)

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Load Pose and Face Mesh models
pose = mp_pose.Pose()
face_mesh = mp_face.FaceMesh()

# Open video file
video_path = './data/kernel-brain-data-jokes.mp4'
cap = cv2.VideoCapture(video_path)

if PLAY_AUDIO:
    # Play audio using FFmpeg in a separate process
    audio_process = subprocess.Popen(["ffplay", "-nodisp", "-autoexit", video_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Create a pose detector
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab a frame.")

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame to detect pose
    pose_results = pose.process(frame_rgb)
    face_results = face_mesh.process(frame_rgb)
    
    # Detect shaking shoulders (pose-based laughter)
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,])
        right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
        shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)
        
        # Detect rapid shoulder movement (simple heuristic)
        if shoulder_distance > 0.02: # Adjust based on testing
            print("Possible laughter detected from shoulder movement")
            
    # Detect Facial Expressions (smiling)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mouth_top = np.array([face_landmarks.landmark[13].x, face_landmarks.landmark[13].y])
            mouth_bottom = np.array([face_landmarks.landmark[14].x, face_landmarks.landmark[14].y])
            mouth_open_ratio = np.linalg.norm(mouth_top - mouth_bottom)
            
            if mouth_open_ratio > 0.015: # Adjust threshold
                print("Possible laughter detected (facial expression)")

    # Draw Landmarks if pose is detected
    # if pose_results.pose_landmarks:
    #     mp_drawing.draw_landmarks(
    #         frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    # if face_results.multi_face_landmarks:
    #     for face_landmarks in face_results.multi_face_landmarks:
    #         mp_drawing.draw_landmarks(
    #             frame, face_landmarks, mp_face.FACEMESH_TESSELATION)
            # mp_drawing.draw_landmarks(
                # frame, face_landmarks, mp_face.FACEMESH_CONTOURS)

    # Create Random Colors
    # POINT_COLOR_R = random.randint(0, 255)
    # POINT_COLOR_G = random.randint(0, 255)
    # POINT_COLOR_B = random.randint(0, 255)
    # Draw a point (small filled red circle @ coordinate 100, 100)
    # cv2.circle(frame, (100, 100), 5, (POINT_COLOR_B, POINT_COLOR_G, POINT_COLOR_R), -1)
    # # Draw random points
    # frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # random_height = random.randint(0, frame_height)
    # random_width = random.randint(0, frame_width)
    # cv2.circle(frame, (random_width, random_height), 5, (POINT_COLOR_B, POINT_COLOR_G, POINT_COLOR_R), -1)
    # Display frame
    cv2.imshow("Real-time Video with audio", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        if PLAY_AUDIO:
            audio_process.terminate()
        break

cap.release()
cv2.destroyAllWindows()
if PLAY_AUDIO:
    audio_process.terminate()
