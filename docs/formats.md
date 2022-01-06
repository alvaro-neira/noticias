## .pb
The .pb format is the protocol buffer (protobuf) format, and in Tensorflow, this format is used to hold models. Protobufs are a general way to store data by Google that is much nicer to transport, as it compacts the data more efficiently and enforces a structure to the data. When used in TensorFlow, it's called a SavedModel protocol buffer, which is the default format when saving Keras/ Tensorflow 2.0 models. More information about this format can be found here and here.

https://stackoverflow.com/questions/51278213/what-is-the-use-of-a-pb-file-in-tensorflow-and-how-does-it-work

## .pbtxt
Also from tensorflow. Represent the graph of the DNN.