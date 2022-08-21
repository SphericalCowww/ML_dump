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
        self.w = [0, 0];
        self.b = 0;

    def fit(self, data):
        self.data = data;
        opt = {};  #[w, b]
#extract featureMax/Min
        data_temp = [];
        for yi in self.data:
            for featureSet in self.data[yi]:
                for feature in featureSet:
                    data_temp.append(feature);
        self.featureMax = max(data_temp);
        self.featureMin = min(data_temp);
        data_temp = None;
#find options such that the constain, -yi(w*xi+b)+1<= 0, is satisfied
        opt = {};  #[w, b]
        stepSet = [self.featureMax*0.1,
                   self.featureMax*0.01,
                   self.featureMax*0.001];
        stepRange = 30;
        bScalingRange = 30; 
        bScaling = 5;
        self.w = [0, 0];
        self.b = 0;
        for step in stepSet:            
            w_scan = np.array([stepRange*step, stepRange*step]);
            while(w_scan[0] > -stepRange*step):
                for b in np.arange(self.b - bScalingRange*bScaling*step,
                                   self.b + bScalingRange*bScaling*step,
                                   bScaling*step):
                        w = (self.w + w_scan);
                        found_option = True;
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i; 
                                if((-1*yi*(np.dot(w, xi) + b) + 1 < 0)) == False:
                                    found_option = False;
                        if(found_option == True):
                            opt[np.linalg.norm(w)] = [w, b];
                if(w_scan[1] > -stepRange*step):
                    w_scan[1] = w_scan[1] - step;
                else:
                    w_scan[1] = stepRange*step;
                    w_scan[0] = w_scan[0] - step;
#find the maximize |w| among the options
            print("w[0] in (" + str((self.w - stepRange*step)[0]) + ", " + str((self.w + stepRange*step)[0]) + ", " + str(step) + ")");
            print("w[1] in (" + str((self.w - stepRange*step)[1]) + ", " + str((self.w + stepRange*step)[1]) + ", " + str(step) + ")");
            print("b in (" + str(self.b - bScalingRange*step) + ", "+ str(self.b + bScalingRange*bScaling*step) + ", " + str(bScaling*bScaling*step) + ")");
            norms = sorted([n for n in opt]);
            opt_choice = opt[norms[0]];
            self.w = opt_choice[0];
            self.b = opt_choice[1];
            print("(w, b) = (" + str(self.w) + ", " + str(self.b)+ ")");
            print("");
    def predict(self, data):
        #sign(w.x + b)
        classification = np.sign(np.dot(self.w, np.array(data)) + self.b);
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
            if(w[1] != 0):
                return ((-w[0]*x - b + v)/w[1]);
            else:
                return 1000*self.featureMax;
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




