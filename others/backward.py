import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import generator
import forward

STEPS = 40000
BATCH_SIZE = 32
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.999
REGULARIZER = 0.01

def backward():
    x = tf.placeholder(tf.float32,shape= (None,2))
    y_ = tf.placeholder(tf.float32,shape= (None,1))
    X, Y_, Y_c = generator.generator()

    pred_y = forward.forward(x,REGULARIZER)
    global_step = tf.Variable(0,trainable=False)
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        300/BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    #定义损失函数
    loss_mse = tf.reduce_mean(tf.square(pred_y-y_))
    loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

    #定义反向传播方法：包括正则化
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(STEPS):
            start = (i*BATCH_SIZE)%300
            end = start + BATCH_SIZE
            sess.run(train_step,feed_dict={x: X[start:end],y_: Y_[start:end]})
            if i % 200 ==0:
                loss_v = sess.run(loss_total,feed_dict={x:X,y_:Y_})
                print("After %d steps, loss_v is %f:"%(i,loss_v))
            xx,yy = np.mgrid[-3:3:0.1,-3:3:0.1]
            grid = np.c_[xx.ravel(),yy.ravel()]
            probs = sess.run(pred_y,feed_dict={x:grid})
            probs = probs.reshape(xx.shape)
        print("xx.shape:",xx.shape)
        print("yy.shape:",yy.shape)
        plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))
        plt.contour(xx,yy,probs,levels=[0.5])

        plt.show()

if __name__ == '__main__':
    backward()




