import time 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward
TEST_INTERVAL_SECS = 5

def test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None,mnist_forward.OUTPUT_NODE])
        y = mnist_forward.forward(x,None)
        
        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        #qiupingjun 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        j = 0
        while j<20:
            #gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
            #with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
#                print("----")
#                print(ckpt)
#                print("----")
                if ckpt and ckpt.model_checkpoint_path:
#                    print("ok?")
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#                    print("global_step:",global_step)
                    accuracy_score = sess.run(accuracy, feed_dict = {x: mnist.test.images[:256], y_: mnist.test.labels[:256]})
                    print("After %s training step(s). test accuracy = %f"%(global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
                time.sleep(TEST_INTERVAL_SECS)
                j += 1
                
def main():
    mnist = input_data.read_data_sets("./MNIST_data/",one_hot = True)
    test(mnist)
    
if __name__ == "__main__":
    main()