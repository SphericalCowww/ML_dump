#uses Geron, pg 67, 236, 237
import numpy as np
from datetime import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.datasets import fetch_california_housing

def fetch_batch(housing_shuffled, batch_index, batch_size):
    sampleN = housing_shuffled.data.shape[0];
    last = (batch_index + 1)*batch_size - 1;
    if last > sampleN - 1:
        last = sampleN - 1;
    X_batch = housing_shuffled[batch_index*batch_size:last, :-1];
    y_batch = housing_shuffled[batch_index*batch_size:last, -1:];
    return X_batch, y_batch;

if __name__ == "__main__":
    n_epochs = 5;
    learning_rate = 0.01;
    batch_size = 100;

    housing = fetch_california_housing();
    sampleN, featureN = housing.data.shape;
    pipe = Pipeline([("imputer", Imputer(strategy="median")),
                     ("std_scaler", StandardScaler())]);
    housing_scaled = pipe.fit_transform(housing.data);
    housing_biased = np.c_[np.ones((sampleN, 1)), housing_scaled.data];
    housing_shuffled = np.c_[housing_biased, housing.target];

    X = tf.placeholder(tf.float32, shape=(None, featureN + 1), name="X");
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y");
    theta = tf.Variable(tf.random_uniform([featureN + 1, 1], -1.0, 1.0), 
                        name="theta");
    y_predict = tf.matmul(X, theta, name="y_predict");
    with tf.name_scope("loss") as scope:
        error = tf.subtract(y_predict, y);
        mse = tf.reduce_mean(tf.square(error), name="mse");
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate);
    training = optimizer.minimize(mse);

    logfilename = "tf_log/run" + datetime.utcnow().strftime("%y%m%d_%H%M%S") + "/";
    mse_summary = tf.summary.scalar("MSE", mse);
    file_writer = tf.summary.FileWriter(logfilename, tf.get_default_graph());
    summary_str = "";

    n_batches = int(np.ceil(sampleN/batch_size));
    step = 0;
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()); 
        print("###############################################");
        for epoch in range(n_epochs):
            np.random.shuffle(housing_shuffled);
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(housing_shuffled, batch_index,
                                               batch_size);
                sess.run(training, feed_dict={X: X_batch, y: y_batch}); 
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch});
                step = epoch*n_batches + batch_index;
                file_writer.add_summary(summary_str, step);
            print("epoch ", epoch, 
                  ": mse = ", mse.eval(feed_dict={X: X_batch, y: y_batch}));
        best_theta = theta.eval();
        print("###############################################");
        print("best theta = ", best_theta);
        file_writer.close();
    print("\nSaving log file:");
    print(logfilename);


