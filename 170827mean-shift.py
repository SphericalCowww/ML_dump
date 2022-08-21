import sys, math, datetime

import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use('ggplot');

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd");

if __name__ == "__main__":
    centers = [[1, 1, 1], [5, 5, 5], [3, 10, 10]];
    X, _ = make_blobs(n_samples=500, centers=centers, cluster_std=1);

    clt = MeanShift();
    clt.fit(X);
    labels = clt.labels_;
    cluster_centers = clt.cluster_centers_;
    n_clusters_ = len(np.unique(labels));
    print(cluster_centers);
    print("Number of estimated clusters: ", n_clusters_);

    colors = ["g", "b", "c", "r", "k"];
    fig = plt.figure();
    ax = fig.add_subplot(111, projection="3d");

    for i in range(len(X)):
        ax.scatter(X[i][0], X[i][1], X[i][2], color=colors[labels[i]], marker="o");
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], marker="x", color="k", s=150, linewidths=5, zorder=10);

    plt.show();    




 
