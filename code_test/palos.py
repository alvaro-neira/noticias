import tensorflow as tf
from tensorflow.keras.models import save_model, Sequential

model_path = r"/Users/aneira/noticias/Gender-and-Age-Detection"

model = tf.keras.models.load_model(model_path)

save_model(model,model_path + r"new_model.h5", save_format='h5')