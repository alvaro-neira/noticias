import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

New_Model = tf.keras.models.load_model('/Users/aneira/noticias/Gender-and-Age-Detection') # Loading the Tensorflow Saved Model (PB)
print(New_Model.summary())