from PIL import Image
from IPython.display import display
from roboflow import Roboflow

# Open and display the original image
original_image = Image.open("/content/download.jpg")
display(original_image)

# Resize the image
new_size = (200, 200)
resized_image = original_image.resize(new_size)
display(resized_image)

# Save the resized image
resized_image.save("output1.png")

# Close the original image
original_image.close()

# Initialize Roboflow
rf = Roboflow(api_key="HMmlhEPsW0NhJ2zbkHAS")

# Get the model from Roboflow
model = rf.workspace().project("car-logo-detector").version(1).model

# Predict on the resized image and save the prediction
prediction = model.predict("/content/output1.png", confidence=40, overlap=30)
prediction.save("prediction5.jpg")
print(prediction.json())
