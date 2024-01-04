import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(image, title):
    histogram, bins = np.histogram(image.ravel(), 256, [0, 256])
    plt.figure()
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Pixel Count")
    plt.plot(histogram, color='black')
    plt.xlim([0, 256])
    plt.show()

def histogram_equalization(image):
    equ = cv2.equalizeHist(image)
    return equ

# Load the input image (replace 'path/to/your/image.jpg' with the actual image file path)
image_path = 'input_image.jpg'
input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Plot the original histogram
plot_histogram(input_image, "Original Histogram")

# Perform histogram equalization
equalized_image = histogram_equalization(input_image)

# Plot the equalized histogram
plot_histogram(equalized_image, "Equalized Histogram")

# Display the original and equalized images
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(input_image, cmap='gray')
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title("Equalized Image")
plt.show()
