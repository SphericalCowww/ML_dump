import sys, os, time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import cv2
import magic
import shutil
import random
import pickle

VERBOSITY = 1;
IMGSIZE = 100;
EPOCHN = 10;
DATADIRNAME = os.getcwd() + "/data/";
FIGDIRNAME = os.getcwd() + "/fig/";

#https://stackoverflow.com/questions/43895750/keras-input-shape-for-conv2d-and-manually-loaded-images


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
    '''
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

    pickle.dump(x_train, open("x_train.pickle", "wb"));
    pickle.dump(y_train, open("y_train.pickle", "wb"));
    pickle.dump(x_test, open("x_test.pickle", "wb"));
    pickle.dump(y_test, open("y_test.pickle", "wb"));
    '''
#loading pickle
    x_train = pickle.load(open("x_train.pickle", "rb"));
    y_train = pickle.load(open("y_train.pickle", "rb"));
    x_test  = pickle.load(open("x_test.pickle", "rb"));
    y_test  = pickle.load(open("y_test.pickle", "rb"));

    x_trainNorm = tf.keras.utils.normalize(x_train, axis=1);
    x_testNorm  = tf.keras.utils.normalize(x_test, axis=1);

    S = x_trainNorm.shape;
    x_trainInput = x_trainNorm.reshape(S[0], S[1], S[2], 1);
    x_testInput  = x_testNorm.reshape(len(x_testNorm), S[1], S[2], 1);
#learning
    '''
    model = tf.keras.models.Sequential();
    model.add(Conv2D(64, kernel_size=(3,3), input_shape=(S[1],S[2],1)));
    model.add(Activation("relu"));
    model.add(MaxPooling2D(pool_size=(2,2)));
    model.add(Conv2D(64, kernel_size=(3,3)));
    model.add(Activation("relu"));
    model.add(MaxPooling2D(pool_size=(2,2)));
    model.add(Flatten());
    model.add(Dense(64));
    model.add(Activation("relu"));
    model.add(Dense(1));
    model.add(Activation("sigmoid"));
    
    model.compile(optimizer="adam",\
                  loss="binary_crossentropy",\
                  metrics=["accuracy"]);
    model.fit(x_trainInput, y_train, epochs=EPOCHN);
    model.save("CatDogConv2D.model");
    '''
#modeling prediction    
    modelO = tf.keras.models.load_model("CatDogConv2D.model");

    loss, acc = modelO.evaluate(x_testInput, y_test);
    pred = modelO.predict([x_testInput]);
    predVals = [];
    if len(categories) == 2:
        for predVal in pred:
            if predVal < 0.5:
                predVals.append(0);
            else:
                predVals.append(1);
    else:
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








 
