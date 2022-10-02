import sys, os, time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


if __name__ == "__main__":
    dataframe = pd.read_csv("housing.txt", delim_whitespace=True, header=None);
    dataset = dataframe.values;
    X = dataset[:,0:13];
    Y = dataset[:,13];

    print(Y);
    sys.exit(0);

    '''
    model = tf.keras.models.Sequential();
    model.add(tf.keras.layers.Flatten());
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu));
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu));
    model.add(tf.keras.layers.Dense(len(set(y_train)), activation=tf.nn.softmax));
    
    model.compile(optimizer="adam",\
                  loss="sparse_categorical_crossentropy",\
                  metrics=["accuracy"]);
    model.fit(x_trainNorm, y_train, epochs=3);
    model.save("mnistNumber.model");
    '''    
    modelO = tf.keras.models.load_model("mnistNumber.model");

    loss, acc = modelO.evaluate(x_testNorm, y_test);
    pred = modelO.predict([x_testNorm]);
    predVals = [];
    for predVal in pred:
        predVals.append(np.argmax(predVal));
#plotting    
    exepath = os.path.dirname(os.path.abspath(__file__));
    print("Saving the following figures:");
    for i, predVal in enumerate(predVals):
        plt.imshow(x_test[i], cmap=plt.cm.binary);
        plt.title("Prediction: "+str(predVal), fontsize=24);
        filenameFig = exepath + "/fig/mnistData"+str(i)+".png";
        plt.savefig(filenameFig, dpi=50);
        plt.close();
        print(filenameFig);










 
