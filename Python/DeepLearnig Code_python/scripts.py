import tensorflow as tf
from tensorflow import keras
from keras import layers
import os
import numpy as np
import pandas as pd
import os, random
from glob import glob
import cv2
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load the Keras model
model_path = "C:/oceanProject_mz/Python/Model/10_31_3.keras"
model = keras.models.load_model(model_path)

# Define the image size expected by the model
expected_image_size = (64, 64)

# Load and preprocess the image
image_path = "C:/ocean/test/mink2.jpg"
img = keras.utils.load_img(image_path, target_size=expected_image_size)
img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch axis

# Make predictions using the model
predictions = model.predict(img_array)
score = float(predictions[0])

# Print the prediction results
print(f"This image is {100 * (1 - score):.2f}% Porpoise and {100 * score:.2f}% Mink.")

# Display the loaded image
plt.imshow(img)
plt.show()