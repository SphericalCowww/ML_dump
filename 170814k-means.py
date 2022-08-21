import sys, math, datetime

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import pickle

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot');

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd");

def handle_nonNum(df):
    columnSet = df.columns.values;
    for column in columnSet:
        text_digit_vals = {};
        def toInt(val):
            return text_digit_vals[val];
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            columnContents = df[column].values.tolist();
            uniqueSet = set(columnContents);
            uniqueIndex = 0;
            for unique in uniqueSet:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = uniqueIndex;
                    uniqueIndex = uniqueIndex + 1;
            df[column] = list(map(toInt, df[column]));      #text_digit_vals used in toInt
    return df;

if __name__ == "__main__":
    df = pd.read_excel("data/titanic.xls");
    df.drop(["body", "name"], 1, inplace=True);
    df.fillna(0, inplace=True); 
    df = handle_nonNum(df);
    print("######################################################################");
    print(df.head());

    X = np.array(df.drop(["survived"], 1).astype(float));
    X = preprocessing.scale(X);
    y = np.array(df["survived"]);

    clf = KMeans(n_clusters=2);
    clf.fit(X);

    correct = 0;
    for i in range(len(X)):
        predict_me = np.array(X[i]);
        predict_me = predict_me.reshape(-1, len(predict_me));
        prediction = clf.predict(predict_me);
        if prediction[0] == y[i]:       #cluster index is arbitrary
            correct = correct + 1;

    print(max(correct/len(X), 1 - correct/len(X)));
        














 
