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
    confidence = 1.0*Counter(dist_sorted).most_common(1)[0][1]/k;
    return result, confidence;

if __name__ == "__main__": 
    data = "breast-cancer-wisconsin.data";
    df = pd.read_csv(data);
    df.replace("?", -99999, inplace=True);
    df.drop(["id"], 1, inplace=True);
    full_data = df.astype(float).values.tolist();
    random.shuffle(full_data);
    print(df.head());
    print("");
    print(df.tail());
    print("");
    print(full_data[:10]);
    print("");
    print(full_data[-10:]);
    print("");
    print("-------------------------------------------------------------------");

    test_size = 0.2;
    train_set = {2:[], 4:[]};
    test_set = {2:[], 4:[]};
    train_data = full_data[:-int(test_size*len(full_data))];
    test_data = full_data[-int(test_size*len(full_data)):];

    for i in train_data:
        train_set[i[-1]].append(i[:-1]);
    for i in test_data:
        test_set[i[-1]].append(i[:-1]);

    total = 0;
    correct = 0;

    for group in test_set:
        for data in test_set[group]:
            result, confidence = k_nearest_neighbors(train_set, data, k = 20);
            if group == result:
                correct += 1;
            else:
                print(confidence);
            total += 1;
    print("Accuracy = " +str(correct) + "/" + str(total) + " = " 
    + str(1.0*correct/total));




