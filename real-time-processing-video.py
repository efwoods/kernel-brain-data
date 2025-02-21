import cv2
import subprocess
import random

PLAY_AUDIO = False
# POINT_COLOR = random.randint(0, 255)

video_path = './data/kernel-brain-data-jokes.mp4'
cap = cv2.VideoCapture(video_path)

if PLAY_AUDIO:
    # Play audio using FFmpeg in a separate process
    audio_process = subprocess.Popen(["ffplay", "-nodisp", "-autoexit", video_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab a frame.")
        break
    
    # POINT_COLOR_R = random.randint(0, 255)
    # POINT_COLOR_G = random.randint(0, 255)
    # POINT_COLOR_B = random.randint(0, 255)
    
    # Draw a point (small filled red circle @ coordinate 100, 100)
    # cv2.circle(frame, (100, 100), 5, (POINT_COLOR_B, POINT_COLOR_G, POINT_COLOR_R), -1)
    
    # Draw random points
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    random_height = random.randint(0, frame_height)
    random_width = random.randint(0, frame_width)
    
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
