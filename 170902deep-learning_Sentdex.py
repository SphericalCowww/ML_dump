import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True);
mnist = input_data.read_data_sets("data/MNIST_data", one_hot=True);
n_nodes_hl1 = 500;
n_nodes_hl2 = 500;
n_nodes_hl3 = 500;

n_classes = 10;
batch_size = 100;
n_epochs = 5;

# input*weight + bias
def NN_model(data):
    hl1  = {"weight": tf.Variable(tf.random_normal([784, n_nodes_hl1])),
            "bias":   tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hl2  = {"weight": tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
            "bias":   tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hl3  = {"weight": tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
            "bias":   tf.Variable(tf.random_normal([n_nodes_hl3]))}
    outl = {"weight": tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
            "bias":   tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hl1["weight"]), hl1["bias"]);
    l1 = tf.nn.relu(l1);            #activation func, e.g. sigmoid
    l2 = tf.add(tf.matmul(l1, hl2["weight"]), hl2["bias"]);
    l2 = tf.nn.relu(l2);
    l3 = tf.add(tf.matmul(l2, hl3["weight"]), hl3["bias"]);
    l3 = tf.nn.relu(l3);
    outl = tf.matmul(l3, outl["weight"]) + outl["bias"];
    return outl;

def train_neural_network(x, y):
    prediction = NN_model(x);
    cost = tf.reduce_mean(\
           tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)); 
    optimizer = tf.train.AdamOptimizer().minimize(cost);    #learning_rate=0.001

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()); 
        for epoch in range(n_epochs):
            epoch_loss = 0;
            for _ in range(int(mnist.train.num_examples/batch_size)):
                x_epoch, y_epoch = mnist.train.next_batch(batch_size);
                _, c = sess.run([optimizer, cost], 
                                feed_dict={x: x_epoch, y: y_epoch});
                epoch_loss += c;
            print("epoch", epoch, ": loss = ", epoch_loss);
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1));
        accuracy = tf.reduce_mean(tf.cast(correct, "float"));
        print("accuracy: ", accuracy.eval({x: mnist.test.images, 
                                           y: mnist.test.labels}));

if __name__ == "__main__":
    x = tf.placeholder("float", [None, 784]);
    y = tf.placeholder("float");
    train_neural_network(x, y);





