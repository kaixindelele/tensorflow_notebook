from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# print(dir(mnist.train.images))
# print(mnist.train.images.shape)

def compute_accuracy(v_xs,v_ys):
	global prediction
	y_pre = sess.run(prediction,feed_dict = {xs:v_xs})
	""" 
	print("y_pre.shape:",y_pre.shape)
	y_pre.shape: (10000, 10)
	tf.argmax(array,0 or 1) can return the index of maxmimum of a variable
	and if array = ([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
	if axis is 0,will compare per row ,so return 4 value
	if axis is 1,will compare per column,so return 3 value
	"""
	correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	result = sess.run(accuracy,feed_dict = {xs:v_xs,ys:v_ys})
	return result



# define placeholder for inputs to network
xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder("float32",[None,10])

# define Weights and biases for network
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# compute prediction without adding hidden layers
prediction = tf.nn.softmax(tf.matmul(xs,W) + b)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

sess = tf.Session()

sess.run(tf.global_variables_initializer())
for i in range(100000):
	batch_xs,batch_ys = mnist.train.next_batch(100)
	sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
	if i % 50 == 0:
		print(compute_accuracy(
			mnist.test.images,mnist.test.labels))




# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)

# print(W)
# print(b)
# for i in range(1000):
# 	batch_xs,batch_ys = mnist.train.next_batch(100)

# 	sess.run(train_step,feed_dict = {x: batch_xs,y_:batch_ys})
