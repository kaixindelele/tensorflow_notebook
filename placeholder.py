import tensorflow as tf 
import numpy as np
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
# t0 = tf.placeholder(tf.float32,[1,2])
# t1 = tf.placeholder(tf.float32,[2,1])
# output = tf.multiply(input1,input2)

# test = np.array([12,12])
# print("test.shape",test.shape)

matrix = tf.matmul(t0,t1)
with tf.Session() as sess:
	print(sess.run(output,feed_dict = {input1:[7.],input2:[3.]}))
	# print(sess.run(output,feed_dict = {t0:[2.,2.],t1:[[1.],[1.]]}))
	# can't display the matrix through placeholder
	