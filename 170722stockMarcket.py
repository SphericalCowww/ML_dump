import sys, math, datetime

import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import pickle

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot');

import quandl

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd");

if __name__ == "__main__":
    df = quandl.get('WIKI/GOOGL');
    df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']];
    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'];
    df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'];
    df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']];
    print("---------------------------------------------------------------Head");

    forecast_days = int(math.ceil(0.01*len(df)));
    df['Forecast'] = df['Adj. Close'].shift(-forecast_days);
    X = np.array(df.drop(['Forecast'], 1));
    y = np.array(df['Forecast']);
    print(df.head());
    print();
    print(df.tail(int(1.5*forecast_days)));
    print();
    print("forecast_days = " + str(forecast_days));
    print("X: ");
    print(X[-int(0.5*forecast_days):]);
    X = preprocessing.scale(X);
    X_forecast = X[-forecast_days:];
    X = X[:-forecast_days];
    y = y[:-forecast_days];
    print("X (processed): ");
    print(X[-int(0.5*forecast_days):]);
    print("y: ");
    print(y[-int(0.5*forecast_days):]);
    print("X_forecast: ");
    print(X_forecast);
    print("-------------------------------------------------------------------");

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2);

###
    clf = LinearRegression(n_jobs=-1);
    #clf = svm.SVR(kernel="poly");
    clf.fit(X_train, y_train);
    with open("stockMarcket.pickle", "wb") as pickle_file:
        pickle.dump(clf, pickle_file);
###   
    pickle_in = open("stockMarcket.pickle", "rb");
    clf = pickle.load(pickle_in);

    accuracy = clf.score(X_test, y_test);
    forecast_set = clf.predict(X_forecast);
    print("accuracy = ", accuracy);
    print("Forecast: ");
    print(forecast_set);

    df['Forecast'] = np.nan;
    last_date = df.iloc[-1].name;
    last_unix = last_date.timestamp();
    one_day = 86400;
    next_unix = last_unix + one_day;
    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix);
        next_unix += one_day;
        df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i];
    print();
    print(df.tail(int(1.5*forecast_days)));

    df['Adj. Close'].plot();
    df['Forecast'].plot();
    plt.legend(loc=4);
    plt.xlabel('Date');
    plt.ylabel('Price');
    plt.show();





