import tensorflow as tf 
state = tf.Variable(0,name='counter')
two = tf.constant(2)
new_value = tf.add(state,two)
update = tf.assign(state,new_value)
init = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(init)
	for _ in range(3):
		sess.run(update)
		print(sess.run(state))