import sys, math, datetime

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pickle

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot');

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd");

if __name__ == "__main__":
    X = np.array([[1, 2],
                  [2, 2],
                  [5, 8],
                  [8, 8],
                  [1, 0],
                  [9, 11]]);
    #plt.scatter(X[:, 0], X[:, 1], s=10, linewidths=5);
    #plt.show();

    clf = KMeans(n_clusters=2);
    clf.fit(X);

    centroids = clf.cluster_centers_;
    labels = clf.labels_;

    colors = ["g.", "b.", "c.", "r.", "k."];


    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=10, linewidths=5);
    for i in range(len(X)):
        plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10);
    plt.show();    




 
