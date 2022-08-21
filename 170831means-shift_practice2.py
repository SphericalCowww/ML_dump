import sys, math, datetime

import pandas as pd
import numpy as np
import pickle
from sklearn.datasets.samples_generator import make_blobs
import random

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot');

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd");

class Mean_Shift:
    def __init__(self, radius=None, radius_norm_step=100):
        self.radius = radius;
        self.radius_norm_step = radius_norm_step;
    def fit(self, data):
        if self.radius == None:
            all_data_centroid = np.average(data, axis=0);
            all_data_norm = np.linalg.norm(all_data_centroid);
            self.radius = all_data_norm/self.radius_norm_step;

        centroidSet = {};
        for i in range(len(data)):
            centroidSet[i] = data[i];
        weightSet = [i for i in range(self.radius_norm_step)][::-1];
        
        while  True:
            new_centroidSet = [];
            for i in centroidSet:
                in_bandwidth = [];
                centroid = centroidSet[i];
                for featureSet in data:
                    distance = np.linalg.norm(featureSet - centroid);
                    if distance == 0:
                       distance = 0.001;
                    weight_index = int(distance/self.radius);
                    if weight_index > self.radius_norm_step - 1:
                        weight_index = self.radius_norm_step - 1;
                    to_add = (weightSet[weight_index])*[featureSet];
                    #to_add = (weightSet[weight_index]**2)*[featureSet];
                    in_bandwidth += to_add;
                new_centroid = np.average(in_bandwidth, axis=0);
                new_centroidSet.append(tuple(new_centroid));
            uniqueSet = sorted(list(set(new_centroidSet)));
            
            to_pop = [];
            for i in uniqueSet:
                for ii in uniqueSet:
                    if i == ii:
                        pass;
                    elif np.linalg.norm(np.array(i) - np.array(ii)) < self.radius:
                        to_pop.append(ii);
                        break;
            for ii in to_pop:
                try:
                    uniqueSet.remove(ii);
                except:
                    pass;                

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
    colors = ["b", "g", "r", "c", "m", "y", "w"];

    X = np.array([[1, 3],
                  [2, 2],
                  [6, 8],
                  [8, 8],
                  [1, 0],
                  [9, 9],
                  [0, 4],
                  [3, 15],
                  [4, 14],
                  [4, 16],]);

    centers = random.randrange(2, 8);
    X, y = make_blobs(n_samples=30, centers=centers, n_features=2);

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

 
