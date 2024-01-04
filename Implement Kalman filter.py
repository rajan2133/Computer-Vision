import cv2
import numpy as np

# Create Kalman Filter
kalman = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, dx, dy), 2 measurement variables (x, y)

# Initial state (position and velocity)
kalman.statePre = np.array([0, 0, 0, 0], dtype=np.float32)

# Transition matrix (describes how the state evolves from one time step to the next)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype=np.float32)

# Measurement matrix (describes how to map the state space to the measurement space)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], dtype=np.float32)

cap = cv2.VideoCapture('C:/Users/Dell/OneDrive/Documents/Computer Vision/LAB/P11/myvideo.mp4')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform face detection
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Take the first detected face (you may want to improve this logic)
        x, y, w, h = faces[0]

        # Predict the next state
        prediction = kalman.predict()

        # Update the Kalman Filter with the measurement (bbox center)
        measurement = np.array([x + w/2, y + h/2], dtype=np.float32)
        kalman.correct(measurement)

        # Get the corrected state (estimated object position)
        estimated_position = prediction[:2]

        # Draw the bounding box and estimated position
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.circle(frame, (int(estimated_position[0]), int(estimated_position[1])), 5, (0, 0, 255), -1)

    cv2.imshow('Object Tracking with Kalman Filter', frame)

    if cv2.waitKey(30) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
