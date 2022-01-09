import tensorflow
import tensorboard
import tensorflow as tf
from tensorflow.python.platform import gfile

print(tensorflow.__version__)
print(tensorboard.__version__)

# Commented out IPython magic to ensure Python compatibility.
MODEL_FILENAME = '/Users/aneira/noticias/Gender-and-Age-Detection/opencv_face_detector_uint8.pb'
LOGDIR = '/Users/aneira/noticias/logs'
# Clear any logs from previous runs
# %cd /content/
# !rm -rf ./logs/


with tf.compat.v1.Session() as sess:
    model_filename = MODEL_FILENAME
    with tf.io.gfile.GFile(model_filename, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.graph_util.import_graph_def(graph_def)
# train_writer = tf.summary.FileWriter(LOGDIR)
# train_writer.add_graph(sess.graph)
# train_writer.flush()
# train_writer.close()
#
# # Commented out IPython magic to ensure Python compatibility.
# # %tensorboard --logdir LOGDIR
#
# train_writer.summary()
