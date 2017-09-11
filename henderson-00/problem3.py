# problem3.py
import numpy as np
import tensorflow as tf

#Multipling the two matrices using numpy
a = np.array([[1,2],[4,-1],[-3,3]])
b = np.array([[-2, 0, 5], [0, -1, 4]])
print(np.dot(a,b))

#Multipling the two matrices using TensorFlow
a = tf.constant(a)
b = tf.constant(b)

#TensorFlow Session
with tf.Session() as sess :
    print(tf.matmul(a,b).eval())
