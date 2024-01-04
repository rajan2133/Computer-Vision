import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the image
image = cv2.imread("input_image.jpg")

# Wait for the user to press a key


# 2. Convert color from RGB to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Grayscale',image)

# 3. Resize 
resized_image = cv2.resize(gray_image, (800, 500))  # Adjust dimensions as desired
#cv2.imshow('res',resized_image)
 

# 4.Rotat
rotation_angle = 50  # Adjust rotation angle as desired
rows, cols = gray_image.shape[:2]
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
rotated_image = cv2.warpAffine(gray_image, M, (cols, rows))
#cv2.imshow('rotat',rotated_image)

b,g,r = cv2.split(image)
#cv2.imshow('split',b)  
#cv2.imshow(g)
#cv2.imshow(r)

merged_image = cv2.merge([r, g, b])
  
# Displaying Merged RGB image
#cv2.imshow('merge',merged_image)
# 5. Display images using matplotlib
fig, axes = plt.subplots(2, 3)
fig.suptitle('Image Operations')

# Original image
axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original')

# Grayscale image
axes[0, 1].imshow(gray_image, cmap='gray')
axes[0, 1].set_title('Grayscale')

# Resized image
axes[0, 2].imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('Resized')

# Rotated image
axes[1, 0].imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('Rotated')

# Split image channels
axes[1, 1].imshow(r, cmap='gray')
axes[1, 1].set_title('Red Channel')

axes[1, 2].imshow(merged_image)
axes[1, 2].set_title('Merged')

# Adjust spacing and display the plot
plt.tight_layout()
plt.show()

cv2.waitKey(0)


