import sys, math, datetime

import pandas as pd
import numpy as np
import pickle
import random

import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

class SVM:
    def __init__(self, vis=True):
        self.vis = vis;
        self.colorSet = {1:"r", -1:"b"};
        if self.vis:
            self.fig = plt.figure();
            self.ax = self.fig.add_subplot(1, 1, 1);
        self.w = 0;
        self.b = 0;

    def fit(self, data):
        self.data = data;
        opt = {};  #[w, b]
        transformSet = [[1 , 1],
                        [-1, 1],
                        [-1, -1],
                        [1 , -1]];

        data_temp = [];
        for yi in self.data:
            for featureSet in self.data[yi]:
                for feature in featureSet:
                    data_temp.append(feature);
        self.featureMax = max(data_temp);
        self.featureMin = min(data_temp);
        data_temp = None;
        
        stepSet = [self.featureMax*0.1,
                   self.featureMax*0.01,
                   self.featureMax*0.001];
        b_range_multiple = 5;
        b_multiple = 5;

        featureRange = self.featureMax*10;
        
        for step in stepSet:
            w = np.array([featureRange, featureRange]);
            optimized = False;
            while(optimized == False):
                for b in np.arange(-1*(self.featureMax*b_range_multiple),
                                   self.featureMax*b_range_multiple,
                                   step*b_multiple):
                    for trans in transformSet:
                        w_trans = w*trans;
                        found_option = True;
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i;
                                if not (yi*(np.dot(w_trans, xi) + b) >= 1):
                                    found_option = False;
                        if(found_option == True):
                            opt[np.linalg.norm(w_trans)] = [w_trans, b];
                if w[0] < 0:
                    optimized = True;
                else:
                    w = w - step;
            print("w in (" + str(0) + ", " + str(featureRange) + ", " + str(step) + ")");
            print("b in (" + str(-1*(self.featureMax*b_range_multiple)) + ", "+ str(self.featureMax*b_range_multiple) + ", " + str(step*b_multiple) + ")");

            norms = sorted([n for n in opt]);
            opt_choice = opt[norms[0]];
            self.w = opt_choice[0];
            self.b = opt_choice[1];
            featureRange = opt_choice[0][0] + step*2;
            print("(w, b) = (" + str(self.w) + ", " + str(self.b)+ ")");
            print("");

    def predict(self, data):
        #sign(w.x + b)
        classification = np.sign(np.dot(np.array(data), self.w) + self.b);
        if classification != 0 and self.vis:
            self.ax.scatter(data[0], data[1], s=200,
                            marker="*", c=self.colorSet[classification]);
        return classification;

    def visualize(self):
        for i in self.data:
            for x in self.data[i]:
                self.ax.scatter(x[0], x[1], s=100, color=self.colorSet[i]);
    #v = w.x + b
        def hyperplane(x, w, b, v):
            return ((-w[0]*x - b + v)/w[1]);
        dataRange = (self.featureMin*0.9, self.featureMax*1.1);
        hyper_x_min = dataRange[0];
        hyper_x_max = dataRange[1];

        psv1 = hyperplane(hyper_x_min, self.w, self.b, 1);
        psv2 = hyperplane(hyper_x_max, self.w, self.b, 1);
        self.ax.plot([hyper_x_min, hyper_x_max], [psv1, psv2], "k");

        nsv1 = hyperplane(hyper_x_min, self.w, self.b, -1);
        nsv2 = hyperplane(hyper_x_max, self.w, self.b, -1);
        self.ax.plot([hyper_x_min, hyper_x_max], [nsv1, nsv2], "k");

        db1 = hyperplane(hyper_x_min, self.w, self.b, 0);
        db2 = hyperplane(hyper_x_max, self.w, self.b, 0);
        self.ax.plot([hyper_x_min, hyper_x_max], [db1, db2], "k--");

        plt.axis([0, 10, 0, 10]);
        plt.show();

if __name__ == "__main__": 
    dataSet = {-1: np.array([[2, 2], [1, 3], [1, 4], [2, 5]]),
               1:  np.array([[6, 5], [7, 7], [7, 4], [6, 6]])};
    dataNew = [2, 7];
    svm = SVM();
    svm.fit(data=dataSet);
    svm.predict(dataNew);
    svm.visualize();




