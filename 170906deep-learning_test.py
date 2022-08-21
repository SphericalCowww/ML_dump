#uses Geron, pg 67, 236, 237
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from datetime import datetime
import numpy as np
import tensorflow as tf

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.datasets import fetch_california_housing

if __name__ == "__main__":
    n_epochs = 1000
    mu = 0.01                   #learning rate

    housing = fetch_california_housing();
    m, n = housing.data.shape;
    housing_withBias = np.c_[np.ones((m, 1)), housing.data];

    pipe = Pipeline([("imputer", Imputer(strategy="median")),
                     ("std_scaler", StandardScaler())]);
    housing_scaled = pipe.fit_transform(housing_withBias);


    X = tf.constant(housing_scaled, dtype=tf.float32, name="X");
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y");

    '''
    XT = tf.transpose(X);
    theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y);
    with tf.Session() as sess:
        theta_val = theta.eval();
        print(theta_val);
    '''

    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta");
    y_predict = tf.matmul(X, theta, name="y_predict");
    error = y_predict - y;
    mse = tf.reduce_mean(tf.square(error), name="mse");
    #grad_mse = tf.gradients(mse, [theta])[0];
    #training = tf.assign(theta, theta - mu*grad_mse);
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=mu);
    training = optimizer.minimize(mse);

    init = tf.global_variables_initializer();

    with tf.Session() as sess:
        sess.run(init);
        print("X = ", X.eval());
        print("y = ", y.eval()); 
        print("###############################################");
        for epoch in range(n_epochs):
            if epoch%100 == 0:
                print("                             Epoch", epoch);
                #print("theta = ", theta.eval());
                #print("y_predict = ", y_predict.eval());
                #print("error = ", error.eval());
                print("mse = ", mse.eval());
                #print("grad_mse = ", grad_mse.eval());
                #print("optimizer = ", optimizer.eval());
                #print("training = ", training.eval());
            sess.run(training);
        best_theta = theta.eval();
        print("###############################################");
        print("best theta = ", best_theta);
        print("y = ", y.eval());
        print("y_predict = ", y_predict.eval());

    print(datetime.utcnow().strftime("%y%m%d_%H%M%S"));






