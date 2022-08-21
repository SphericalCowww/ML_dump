import sys, math, datetime

import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot');

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd");

class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=100):
        self.k = k;
        self.tol = tol;
        self.max_iter = max_iter;
    def fit(self, data):
        self.centroidSet = {};
        for k in range(self.k):
            self.centroidSet[k] = data[k];      #start with an arbitrary data set
        for i in range(self.max_iter):
            self.classificationSet = {};
            for k in range(self.k):
                self.classificationSet[k] = [];
            for featureSet in data:
                distanceSet = [np.linalg.norm(featureSet - self.centroidSet[centroid]) for centroid in self.centroidSet];
                classification = distanceSet.index(min(distanceSet));
                self.classificationSet[classification].append(featureSet);

                #print(i, ":", featureSet);
                #print(self.centroidSet)
                #print(distanceSet);
                #print(self.classificationSet);
                #print();

            prev_centroidSet = dict(self.centroidSet);
            for classification in self.classificationSet:
                self.centroidSet[classification] = np.average(self.classificationSet[classification], axis=0);      #why??
            optimized = True;
            for centroid in self.centroidSet:
                orig_centroid = prev_centroidSet[centroid];
                curr_centroid = self.centroidSet[centroid];

                #print("#########################################");
                #print(centroid, ":");
                #print(orig_centroid);
                #print(curr_centroid);
                #print("#########################################");
                #print();

                if np.sum((curr_centroid - orig_centroid)/orig_centroid*100.0) > self.tol:
                    optimized = False;
            if optimized == True:
                break;
    def predict(self, data):
        distanceSet = [np.linalg.norm(data - self.centroidSet[centroid]) for centroid in self.centroidSet];
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
                  [0, 4]]);
    plt.scatter(X[:, 0], X[:, 1], marker="x", color=colors[0], s=100, linewidths=5);
    plt.show();

    clf = K_Means();#max_iter=1);
    clf.fit(X);
    for centroid in clf.centroidSet:
        plt.scatter(clf.centroidSet[centroid][0], clf.centroidSet[centroid][1],
                    marker="*", color="k", s=100, linewidths=5);
    for classification in clf.classificationSet:
        for featureSet in clf.classificationSet[classification]:
            plt.scatter(featureSet[0], featureSet[1],
                        marker="x", color=colors[classification], s=120, linewidths=5);
    
    dataNew = np.array([[1, 3], 
                        [8, 9],
                        [2, 3],
                        [2, 7],
                        [5, 5]]);
    for data in dataNew:
        classification = clf.predict(data);
        plt.scatter(data[0], data[1],
                    marker="o", color=colors[classification], s=100, linewidths=5);
    plt.show();



 
