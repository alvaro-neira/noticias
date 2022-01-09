# import the necessary library
import tensorflow as tf

# create the x and y variables
# x is given an initial value of 1
x = tf.get_variable('x', dtype=tf.int32, initializer=tf.constant([1]))
# y is given an initial value of 2
y = tf.get_variable('y', dtype=tf.int32, initializer=tf.constant([2]))

# create the constant
c = tf.constant([15], name='constant')
two = tf.constant([2], name='two')

# create the function
function = tf.pow(x, two) + tf.pow(y, two) + tf.multiply(x, y) + tf.multiply(two, x) - c

# create an initializer
init = tf.global_variables_initializer()

# create a session
with tf.Session() as sess:
    # initialize the x and y variable
    init.run()
    # create a file that stores the summary of the operation
    writer = tf.summary.FileWriter("output", sess.graph)
    # run the session
    result = function.eval()
# print the result
print(result)
