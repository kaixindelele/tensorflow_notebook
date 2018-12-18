import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data
import mnist_lenet_forward
import numpy as np
import os

REGULARIZER = 0.0001
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./lenet_model/"
MODEL_NAME = 'lenet'
STEPS = 20000


def backward(mnist):
    x = tf.placeholder(tf.float32,[
        BATCH_SIZE,
        mnist_lenet_forward.IMAGE_SIZE,
        mnist_lenet_forward.IMAGE_SIZE,
        mnist_lenet_forward.NUM_CHANNELS
    ])
    y_ = tf.placeholder(tf.float32,[None,mnist_lenet_forward.OUTPUT_SIZE])
    
    y = mnist_lenet_forward.forward(x,True, regularizer=REGULARIZER)
    
    global_step =  tf.Variable(0,trainable=False)
    
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection("losses"))
    
    learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            mnist.train.num_examples/BATCH_SIZE,
            LEARNING_RATE_DECAY
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step,ema_op]):
        train_op = tf.no_op(name="train")
        
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            save.restore(sess,ckpt.model_checkpoint_path)
            
        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshape_xs = np.reshape(xs,(BATCH_SIZE,
                                     mnist_lenet_forward.IMAGE_SIZE,
                                     mnist_lenet_forward.IMAGE_SIZE,
                                     mnist_lenet_forward.NUM_CHANNELS))
            _, loss_value, step, = sess.run([train_op,loss,global_step],feed_dict={x:reshape_xs, y_:ys} )
            if i % 100 == 0:
                print("After %d steps,the loss is %f"%(step,loss_value))
                saver.save(sess,  os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step = global_step)
                
            
def main():
    mnist = input_data.read_data_sets("./MNIST_data/",one_hot=True)
    backward(mnist)
    
if __name__ == "__main__":
    main()
                
            
        
    
