import tensorflow as tf 
import numpy as np 

x_data = np.float32(np.random.rand(2,100))

#x_data.shape is (2,100)
#np.dot() is the multiply of matrix
#and * is that figure multiply one by one
# original function is y = [0.1,0.2] * [[100个],[100个]]+0.3
# so y is just a figure
y_data = np.dot([0.100,0.200],x_data)+0.300

b0 = tf.Variable(tf.zeros([20,1]))
W0 = tf.Variable(tf.random_uniform([20,2],-1.0,1.0))

y0 = tf.matmul(W0,x_data) + b0

b1 = tf.Variable(tf.zeros([1]))
W1 = tf.Variable(tf.random_uniform([1,20],-1.0,1.0))

y = tf.matmul(W1,y0) + b1


loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for step in range(0,5001):
		sess.run(train)
		if step % 20 == 0:
			print(step,"w0:",sess.run(W0),"b0:",sess.run(b0))
			print(step,"w1:",sess.run(W1),"b1:",sess.run(b1))

