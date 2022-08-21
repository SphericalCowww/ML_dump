import sys, math, datetime

import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pickle

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot');

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd");

if __name__ == "__main__":
    data = "breast-cancer-wisconsin.data";
    df = pd.read_csv(data);
    df.replace("?", -99999, inplace=True);
    df.drop(["id"], 1, inplace=True);
    print(df.head());
    print();
    print(df.tail());
    print("-------------------------------------------------------------------");

    X = np.array(df.drop(["class"], 1));
    y = np.array(df["class"]);

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2);
    
    clf = neighbors.KNeighborsClassifier();
    clf.fit(X_train, y_train);

    accuracy = clf.score(X_test, y_test);
    print("accuracy = " + str(accuracy));
   
    example = np.array([[4, 2, 1, 1, 2, 2, 3, 2, 1], [4, 2, 5, 4, 2, 5, 3, 3, 3]]);
    example = example.reshape(len(example), -1);
    prediction = clf.predict(example);
    print("prediction = " + str(prediction));






 
