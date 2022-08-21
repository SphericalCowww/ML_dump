#uses Geron, pg 67, 236, 237
import numpy as np
from datetime import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.datasets import fetch_california_housing

def fetch_batch(epoch, batch_index, batch_size):
    pass;

if __name__ == "__main__":
    n_epochs = 1000;
    learning_rate = 0.01;

    housing = fetch_california_housing();
    sampleN, featureN = housing.data.shape;
    pipe = Pipeline([("imputer", Imputer(strategy="median")),
                     ("std_scaler", StandardScaler())]);
    housing_scaled = pipe.fit_transform(housing.data);
    housing_biased = np.c_[np.ones((sampleN, 1)), housing_scaled.data];

    X = tf.constant(housing_biased, dtype=tf.float32, name="X");
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y");
    #X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X");
    #y = tf.placeholder(tf.float32, shape=(None, 1), name="y");
    theta = tf.Variable(tf.random_uniform([featureN + 1, 1], -1.0, 1.0), 
                        name="theta");
    y_predict = tf.matmul(X, theta, name="y_predict");
    error = tf.subtract(y_predict, y);
    mse = tf.reduce_mean(tf.square(error), name="mse");
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate);
    training = optimizer.minimize(mse);

    logfilename = "tf_log/run" + datetime.utcnow().strftime("%y%m%d_%H%M%S") + "/";
    mse_summary = tf.summary.scalar("MSE", mse);
    file_writer = tf.summary.FileWriter(logfilename, tf.get_default_graph());
    summary_str = "";

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());
        print("X = ", X.eval());
        print("y = ", y.eval()); 
        print("###############################################");
        for epoch in range(n_epochs):
            if epoch%10 == 0:
                print("epoch ", epoch, ": mse = ", mse.eval());
                summary_str = mse_summary.eval();
                file_writer.add_summary(summary_str, epoch);
            sess.run(training);
        best_theta = theta.eval();
        print("###############################################");
        print("best theta = ", best_theta);
        print("y = ", y.eval());
        print("y_predict = ", y_predict.eval());
        file_writer.close();
    print("\nSaving log file:");
    print(logfilename);


