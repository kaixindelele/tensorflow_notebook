import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE  = 10
LAYER_NODE = 500

def get_weight(shape,regularizer):
    #beijieduande biaozhuncha buhuichaoguo liangge biaozhuncha
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer != None:
        tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w
    
def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def forward(x, regularizer):
    w1 = get_weight((INPUT_NODE,LAYER_NODE),regularizer)
    b1 = get_bias(LAYER_NODE)
    y1 = tf.nn.relu(tf.matmul(x,w1)+b1)
    
    w2 = get_weight((LAYER_NODE,OUTPUT_NODE),regularizer)
    b2 = get_bias(OUTPUT_NODE)
    y2 = tf.matmul(y1,w2)+b2
    return y2

def main():
    x = np.random.random((3,784))
    x = x.astype(np.float32)
    print("x.shape:",x.shape)
    regularizer = 0.001
    print(x.dtype)

    pred_y = forward(x,regularizer)
    print("pred_y:",pred_y[:])
if __name__=="__main__":
    main()