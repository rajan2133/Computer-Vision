import cv2

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the image
img = cv2.imread('C:/Users/Dell/OneDrive/Documents/Computer Vision/LAB/P9/team.jpg')
cv2.imshow('Orignal Image', img)
plt.show()
# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# Draw rectangles around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the result
cv2.imshow('Detected Faces', img)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()



# Display the original and detected faces using matplotlib
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# Original image
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('Original Image')
ax1.axis('off')
# Detected faces image # create a copy to preserve the original
for (x, y, w, h) in faces:
 cv2.rectangle(detected_faces_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
ax2.imshow(cv2.cvtColor(detected_faces_img, cv2.COLOR_BGR2RGB))

ax2.set_title('Detected Faces')
ax2.axis('off')
plt.show()