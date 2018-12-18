# coding:utf-8
# 整个的学习率算法：定义基础学习率，衰减率(decay_rate)和更新步数(已经训练的步数/延迟步长，如果decay_steps不为1，很明显会产生梯度)。
# 新学习率的公式 : decayed_learning_rate = learning_rate*(decay_rate^(global_steps/decay_steps)
# 

# 设损失函数 loss = (W + 1) ^ 2,令W初值是常数10。反向传播就是求最优W，即求最小loss对应的W值。
# 使用指数衰减的学习率，在迭代初期得到较高的下降速度，可以在较小的训练轮数下取得更有收敛度

import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np

LEARNING_RATE_BASE = 0.3 #最初的学习率
LEARNING_RATE_DECAY = 0.99 #学习衰减率
LEARNING_RATE_STEP = 1 #喂入多少轮BATCH_SIZE后，更新一次学习率，一般设为：总样本数/BATCH_SIZE

# 运行了几轮BATCH_SIZE的计数器，初值给0，设为不被训练
global_step = tf.Variable(0,trainable = False)
# 定义指数下降学习率，没用自己的公式，而是调用如下函数
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,LEARNING_RATE_STEP,LEARNING_RATE_DECAY,staircase = True)



W = tf.Variable(tf.constant(5,dtype = tf.float32))
loss = tf.square(W + 1)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)

with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	w_val = []
	loss_val = []

	for i in range(400):
		sess.run(train_step)
		# 训练的时候需要更新learning_rate，并且获取rate值
		learning_rate_val = sess.run(learning_rate)
		# global_step竟然也能在会话中更新？
		global_step_val = sess.run(global_step)
		w_val.append(sess.run(W))
		loss_val.append(sess.run(loss))
		print("After %s steps: learning_rate_val is %f,\t global_step_val is %f ." 
			% (i,learning_rate_val,global_step_val))
		# print("After %s steps: W is %f,\t loss is %f ." % (i,w_val,loss_val))
	print("len(w_val):",len(w_val))
	length = 40
	plt.plot(range(length),w_val[:length],linestyle = '--', color = "b",label = "w_val")
	plt.scatter(range(length),w_val[:length],s = 50,color = "b")
	plt.plot(range(length),loss_val[:length],color = "r",label = "loss_val")
	plt.scatter(range(length),loss_val[:length],s = 50,color = "r")
	plt.legend(loc = "best")
	plt.show()
