import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.core.framework import graph_pb2


def load_pb(pb_file_path):
    with session.Session(graph=ops.Graph()) as sess:
        with gfile.FastGFile(pb_file_path, 'rb') as f:
            graph_def = graph_pb2.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

        # print(sess.run('b:0'))
        # #
        # input_x = sess.graph.get_tensor_by_name('x:0')
        # input_y = sess.graph.get_tensor_by_name('y:0')
        # #
        # op = sess.graph.get_tensor_by_name('op_to_store:0')
        # #
        # ret = sess.run(op, {input_x: 3, input_y: 4})
        # print(ret)


load_pb('/Users/aneira/noticias/Gender-and-Age-Detection/opencv_face_detector_uint8.pb');