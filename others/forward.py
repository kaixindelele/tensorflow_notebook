#coding:utf-8
#导入模块，生成模拟数据集
import tensorflow as tf 
import generator
import numpy as np
# 定义网络输入、参数和输出，定义前向传播过程
def get_weight(shape, regularizer):
	W = tf.Variable(tf.random_normal(shape),dtype = tf.float32)
	
	tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(W))
	return W

def get_bias(shape):
	b = tf.Variable(tf.constant(0.01,shape = shape))
	return b

def forward(X,regularizer):
	W1 = get_weight([2,11],regularizer)
	b1 = get_bias([11])
	y1 = tf.nn.relu(tf.matmul(X,W1) + b1)

	W2 = get_weight([11,1],regularizer)
	b2 = get_bias([1])
	y =tf.matmul(y1,W2) +b2 #输出层不过激活

	return y


if __name__ == '__main__':
    X,Y,Y_c = generator. generator()
    X = np.array(X)
    Y = np.array(Y)
    Y_c = np.array(Y_c)
    print("X:",X.shape)
    print("Y:",Y.shape)
    print("Y_c:",Y_c.shape)
