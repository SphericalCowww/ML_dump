import sys, os, time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import magic
import shutil
import random
import pickle

VERBOSITY = 1;
IMGSIZE = 100;
EPOCHN = 8;
DATADIRNAME = os.getcwd() + "/data/";
TESTDIRNAME = os.getcwd() + "/data/ztest/";
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
#load test data
    categories = ["cat", "dog"]; 
    dataTest = [];
    for i, y in enumerate(categories):
        dirPath = TESTDIRNAME + y + "/";
        if len(os.listdir(dirPath)) < 10:
            dataRename(dirPath);
        for imgName in os.listdir(dirPath):
            try:
                origImgFile = cv2.imread(dirPath+"/"+imgName);
                resizedImgFile = cv2.resize(origImgFile, (IMGSIZE, IMGSIZE));
                dataTest.append([resizedImgFile, i, dirPath+"/"+imgName]);
            except Exception as e:
                pass;
    imgsTest = [];
    pathsTest = [];
    for x, y, p in dataTest:
        imgsTest.append(x);
        pathsTest.append(p);
    imgsTest = np.array(imgsTest, dtype=np.float32);
    batchSize, height, width, channelN = imgsTest.shape;
#filtering
    filters = np.zeros(shape=(7, 7, channelN, 2), dtype=np.float32);
    filters[:, 3, :, 0] = 1; #vertical filter
    filters[3, :, :, 1] = 1; #horizontal filter
   
    imgsOut = tf.nn.conv2d(imgsTest, filters, strides=1, padding="SAME");
#plotting 
    exepath = os.path.dirname(os.path.abspath(__file__));
    print(imgsOut.shape);
    print("Saving the following figures:");
    for i, image in enumerate(imgsTest):
        filenameFig = exepath + "/fig/DogOrCat"+str(i)+".png";
        cv2.imwrite(filenameFig, image);
        print(filenameFig);
    for i, image in enumerate(imgsOut[:, :, :, 0]):
        plt.imshow(image, cmap="gray");
        filenameFig = exepath + "/fig/DogOrCatV"+str(i)+".png";
        plt.savefig(filenameFig, dpi=50);
        plt.close();
        print(filenameFig);
    for i, image in enumerate(imgsOut[:, :, :, 1]):
        plt.imshow(image, cmap="gray");
        filenameFig = exepath + "/fig/DogOrCatH"+str(i)+".png";
        plt.savefig(filenameFig, dpi=50);
        plt.close();
        print(filenameFig);





 
