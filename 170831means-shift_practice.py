import sys, math, datetime

import pandas as pd
import numpy as np
import pickle
from sklearn.datasets.samples_generator import make_blobs

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot');

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd");

class Mean_Shift:
    def __init__(self, radius=5):
        self.radius = radius;
    def fit(self, data):
        centroidSet = {};
        for i in range(len(data)):
            centroidSet[i] = data[i];
        while True:
            new_centroidSet = [];
            for i in centroidSet:
                in_bandwidth = [];
                centroid = centroidSet[i];
                for featureSet in data:
                    if np.linalg.norm(featureSet - centroid) < self.radius:
                        in_bandwidth.append(featureSet);
                new_centroid = np.average(in_bandwidth, axis=0);
                new_centroidSet.append(tuple(new_centroid));
            uniqueSet = sorted(list(set(new_centroidSet)));
            
            prev_centroidSet = dict(centroidSet);
            centroidSet = {};

            for i in range(len(uniqueSet)):
                centroidSet[i] = np.array(uniqueSet[i]); 
            optimized = True;
            for i in centroidSet:
                if np.array_equal(centroidSet[i], prev_centroidSet[i]) == False:
                    optimized = False;
                    #break;
            if optimized == True:
                break;
        self.centroidSet = centroidSet;    
    
        self.classificationSet = {};
        for i in range(len(self.centroidSet)):
            self.classificationSet[i] = [];
        for featureSet in data:
            distanceSet = [np.linalg.norm(featureSet - self.centroidSet[centroid])
                           for centroid in self.centroidSet];
            classification = distanceSet.index(min(distanceSet));
            self.classificationSet[classification].append(featureSet);
    def predict(self, data):
        for featureSet in data:
            distanceSet = [np.linalg.norm(featureSet - self.centroidSet[centroid])
                           for centroid in self.centroidSet];
            classification = distanceSet.index(min(distanceSet));
        return classification;

if __name__ == "__main__":
    colors = ["g", "b", "c", "r", "k"];

    X = np.array([[1, 2],
                  [2, 2],
                  [5, 8],
                  [8, 8],
                  [1, 0],
                  [9, 9],
                  [0, 4],
                  [3, 15],
                  [4, 14],
                  [4, 16],]);

    X, y = make_blobs(n_samples=50, centers=3, n_features=2);

    #plt.scatter(X[:, 0], X[:, 1], marker="o", color=colors[0], s=10, linewidths=5);
    #plt.show();

    clf = Mean_Shift();
    clf.fit(X);
    centroidSet = clf.centroidSet;

    for classification in clf.classificationSet:
        for featureSet in clf.classificationSet[classification]:
            plt.scatter(featureSet[0], featureSet[1], marker="*",
                        color=colors[classification], s=10, linewidths=5)

    for c in centroidSet:
            plt.scatter(centroidSet[c][0], centroidSet[c][1], marker="x", color="k", 
                        s=100, linewidths=5);
    plt.show();

 
