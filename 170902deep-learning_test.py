import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


x1 = tf.Variable(3, name="x1");
x2 = tf.Variable(4, name="x2");
f = tf.multiply(x1, x2);
print(f);

init = tf.global_variables_initializer();

with tf.Session() as sess:
    #x1.initializer.run();
    #x2.initializer.run();
    init.run();
    result1 = sess.run(f);
    result2 = f.eval();
    print(result1);
    print(result2);



