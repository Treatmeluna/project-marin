import train
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras import layers
import os
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy 
import pandas 


img = keras.utils.load_img(
    "C:/ocean/Minke whale_89.png", target_size=train.image_size  # 파일 경로 설정 
)
plt.imshow(img)

img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = train.model.predict(img_array)
score = float(predictions[0])
print(f"This image is {100 * (1 - score):.2f}% Porpoise and {100 * score:.2f}% Mink.") 