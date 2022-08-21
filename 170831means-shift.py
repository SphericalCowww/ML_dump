import sys, math, datetime

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.cluster import MeanShift
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
    original_df = pd.DataFrame.copy(df);
    df.drop(["body", "name"], 1, inplace=True);
    df.fillna(0, inplace=True); 
    df = handle_nonNum(df);
    print("######################################################################");
    print(df.head());

    X = np.array(df.drop(["survived"], 1).astype(float));
    X = preprocessing.scale(X);
    y = np.array(df["survived"]);

    clf = MeanShift();
    clf.fit(X);

    labelSet = clf.labels_;
    n_clusters = len(np.unique(labelSet));
    cluster_centers = clf.cluster_centers_;
    
    original_df["cluster_group"] = np.nan;

    for i in range(len(X)):
        original_df["cluster_group"].iloc[i] = labelSet[i];

    survival_rates = {};
    for i in range(n_clusters):
        temp_df = original_df[ (original_df["cluster_group"] == i) ];
        survival_cluster = temp_df[ (temp_df["survived"] == 1) ];
        survival_rates[i] = len(survival_cluster)/len(temp_df);

    print("######################################################################");
    print(survival_rates);
    for i in range(n_clusters):
        print("################################ ", i);
        print(original_df[( original_df["cluster_group"] == i) ].head());
        print();
        print(original_df[( original_df["cluster_group"] == i) ].describe());











 
