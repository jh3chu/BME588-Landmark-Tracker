from http.client import GATEWAY_TIMEOUT
import cv2
from matplotlib.figure import Figure
from matplotlib.pyplot import close
import mediapipe as mp
import numpy as np
import time
import statistics
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

keypoints = []
PoI = ["NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR", "LEFT_SHOULDER", "RIGHT_SHOULDER", "MOUTH_LEFT", "MOUTH_RIGHT"]
time_tracker = []

# Setup Mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        start=time.time()
        lndmrk = np.zeros(27)
        ret, img = cap.read()

        # Recolour to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False # Save memory before passing to pose estimation

        # Make detection
        results = pose.process(img)

        # Recolour back to BGR
        img.flags.writeable = True 
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Extract joints
        try:
            landmarks = results.pose_world_landmarks.landmark
            
            # Append landmarks of interest (Note - coords are multipled my -1)
            for i, point in enumerate(PoI):
                lndmrk[3*i]= (landmarks[getattr(mp_pose.PoseLandmark,point).value].x)
                lndmrk[3*i+1]=(landmarks[getattr(mp_pose.PoseLandmark,point).value].y)*-1
                lndmrk[3*i+2]=(landmarks[getattr(mp_pose.PoseLandmark,point).value].z)*-1
            keypoints.append(lndmrk)          
        except:
            pass

        # Render detections
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
        cv2.imshow('Video Feed', img)
        end=time.time()
        time_tracker.append(end - start)
        # print(end - start) 

        # Plot landmarks
        # mp_drawing.plot_landmarks(results.pose_world_landmarks,  mp_pose.POSE_CONNECTIONS)
        # close(1)
        
        if cv2.waitKey(1) == 113: # Press q to quit
            break

np.savetxt("landmark_tracker.csv", np.array(keypoints), delimiter=",",
    header = ",".join(["Nose_x", "Nose_y", "Nose_z", "L_Eye_x", "L_Eye_y", "L_Eye_z",\
             "R_Eye_x", "R_Eye_y", "R_Eye_z", "L_Ear_x", "L_Ear_y", "L_Ear_z", "R_ear_x", "R_ear_y", "R_ear_z",\
             "L_Shoulder_x", "L_Shoulder_y", "L_Shoulder_z", "R_Shoulder_x", "R_Shoulder_y", "R_Shoulder_z",\
             "L_Mouth_x", "L_Mouth_y", "L_Mouth_z", "R_Mouth_x", "R_Mouth_y", "R_Mouth_z"]))
del time_tracker[0]
print(statistics.mean(time_tracker))
cap.release()
cv2.destroyAllWindows()