import sys, os, time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import magic
import shutil
import random

VERBOSITY = 1;
IMGSIZE = 100;
DATADIRNAME = os.getcwd() + "/data/";
FIGDIRNAME = os.getcwd() + "/fig/";

def dataRename(dirPath):
    dirPathFull = dirPath + "full/";
    fileList = os.listdir(dirPathFull);
    index = 0;
    if VERBOSITY >= 1:
        print("Renaming files in " + dirPath);
    for origFileName in fileList:
        origFileName = dirPathFull + origFileName;
        fileType = magic.from_file(origFileName, mime=True).split("/")[-1];
        if fileType == "jpeg":
            newFileName = dirPath + str(index) + ".jpg";
            shutil.copy2(origFileName, newFileName);
            index += 1;
if __name__ == "__main__":
#input images
    categories = ["dog", "cat"];
    dataTrain = [];
    for i, y in enumerate(categories):
        dirPath = DATADIRNAME + y + "/";
        if len(os.listdir(dirPath)) < 10:
            dataRename(dirPath);
        for imgName in os.listdir(dirPath):
            try:
                origImgFile = cv2.imread(dirPath+"/"+imgName, cv2.IMREAD_GRAYSCALE);
                residedImgFile = cv2.resize(origImgFile, (IMGSIZE, IMGSIZE));
                dataTrain.append([residedImgFile, i]);
            except Exception as e:
                pass;
    random.shuffle(dataTrain);
#data management
    x_train = [];
    y_train = [];
    x_test = [];
    y_test = [];
    for x, y in dataTrain:
        x_train.append(x);
        y_train.append(y);
        x_test.append(x);
        y_test.append(y);
    x_train = np.array(x_train);
    y_train = np.array(y_train);
    x_test = np.array(x_test);
    y_test = np.array(y_test);

    x_trainNorm = tf.keras.utils.normalize(x_train, axis=1); 
    x_testNorm = tf.keras.utils.normalize(x_test, axis=1);
#model setup 
    model = tf.keras.models.Sequential();
    model.add(tf.keras.layers.Flatten());
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu));
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu));
    model.add(tf.keras.layers.Dense(len(set(y_train)), activation=tf.nn.softmax));
#modeling
    model.compile(optimizer="adam",\
                  loss="sparse_categorical_crossentropy",\
                  metrics=["accuracy"]);
    model.fit(x_trainNorm, y_train, epochs=5);
    model.save("CatDogBasic.model");
#modeling prediction    
    modelO = tf.keras.models.load_model("CatDogBasic.model");

    loss, acc = modelO.evaluate(x_testNorm, y_test);
    pred = modelO.predict([x_testNorm]);
    predVals = [];
    for predVal in pred:
        predVals.append(np.argmax(predVal));
#plotting 
    exepath = os.path.dirname(os.path.abspath(__file__));
    print("Saving the following figures:");
    for i, predVal in enumerate(predVals):
        print(i, predVal);
        plt.imshow(x_test[i], cmap=plt.cm.binary);
        plt.title("Prediction: "+str(categories[predVal]), fontsize=24);
        filenameFig = exepath + "/fig/DogOrCat"+str(i)+".png";
        plt.savefig(filenameFig, dpi=50);
        print(filenameFig);
        plt.close();









 
