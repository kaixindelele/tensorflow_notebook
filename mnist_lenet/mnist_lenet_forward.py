import tensorflow as tf
IMAGE_SIZE = 28
NUM_CHANNELS = 1
CONV1_SIZE = 5
CONV1_KERNEL_NUM = 32
CONV2_SIZE = 5
CONV2_KERNEL_NUM = 64
FC_SIZE = 512
OUTPUT_SIZE = 10

def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
#     print("w:",sess.run(w))
    if regularizer!=None:
        tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w
    
    
def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def conv2d(x,w):
    return tf.nn.conv2d(x,w,[1,1,1,1],padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1] ,strides = [1,2,2,1],padding = "SAME")

def forward(x,train,regularizer):
    conv1_w = get_weight([CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_KERNEL_NUM],regularizer=0.0001)
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    conv1 = conv2d(x,conv1_w)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_b))
    pool1 = max_pool_2x2(relu1)
    
    conv2_w = get_weight([CONV2_SIZE,CONV2_SIZE,CONV1_KERNEL_NUM,CONV2_KERNEL_NUM],regularizer=0.0001)
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1,conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_b))
    pool2 = max_pool_2x2(relu2)
    
    pool_shape = pool2.get_shape().as_list()
    #pool_shape[0]是一个batch的值
    print("pool_shape[0]:",pool_shape[0])
    nodes = pool_shape[1]* pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2,[pool_shape[0],nodes])
    
    fc1_w = get_weight([nodes,FC_SIZE],regularizer=0.0001)
    fc1_b = get_bias([FC_SIZE])
    fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_w)+fc1_b)
    if train: fcl = tf.nn.dropout(fc1,0.5)
        
    fc2_w = get_weight([FC_SIZE,10],regularizer=0.0001)
    fc2_b = get_bias([10])
    y = tf.matmul(fc1,fc2_w)+fc2_b
    return y