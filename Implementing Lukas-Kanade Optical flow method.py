import cv2
import numpy as np

# Load GIF
gif_path = 'C:/Users/Dell/OneDrive/Documents/Computer Vision/LAB/P5/Wm2g.gif'

# Check if the GIF can be loaded successfully
cap = cv2.VideoCapture(gif_path)
if not cap.isOpened():
    print("Error: Could not open GIF.")
    exit()

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Read the first frame
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Get good features to track in the first frame
p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Lucas-Kanade method
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        frame = cv2.circle(frame, (int(c), int(d)), 5, (0, 255, 0), -1)
        frame = cv2.line(frame, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)

    # Display the resulting frame in a new window for each frame
    cv2.imshow('Optical Flow', frame)

    # Break the loop
    key = cv2.waitKey(0)

    # Update previous frame and previous points
    prev_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

# Release the GIF capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
