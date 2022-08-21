import sys, math, datetime

import pandas as pd
import numpy as np
import pickle
import random

import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
style.use("fivethirtyeight")

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn("the number of data point is smaller than k.");
    dists = [];
    for feature in data:
        for point in data[feature]:
            dist = np.linalg.norm(np.array(point) - np.array(predict));
            dists.append([dist, feature]);
    dist_sorted = [];
    for dist in sorted(dists)[:k]:
        dist_sorted.append(dist[1]);
    result = Counter(dist_sorted).most_common(1)[0][0];
    return result;

if __name__ == "__main__": 
    dataset = {"black":[[1, 2], [2, 3], [3, 1], [0, 1], [9, 0]], 
               "red":[[6, 5], [7, 7], [8, 3], [10, 10], [2, 8]]};
    loc = [random.random()*10, random.random()*10]; 
    result = k_nearest_neighbors(dataset, loc);

    for i in dataset:
        for ii in dataset[i]:
            [[plt.scatter(ii[0], ii[1], s=100, color=i)]]
    plt.scatter(loc[0], loc[1], color=result);
    plt.show();




