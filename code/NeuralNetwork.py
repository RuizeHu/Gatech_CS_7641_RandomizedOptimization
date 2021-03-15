import mlrose
import numpy as np
import matplotlib.pyplot as plt
import time

import tests as tests
import numpy as np
import pandas as pd
import time
import gc
import random
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn import tree
from sklearn.metrics import plot_roc_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


class Data():

    def dataAllocation(self, path):
        # Separate out the x_data and y_data and return each
        # args: string path for .csv file
        # return: pandas dataframe, pandas series
        # -------------------------------
        # ADD CODE HERE
        df = pd.read_csv(path)
        xcols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
        ycol = ['y']
        x_data = df[xcols]
        y_data = df[ycol]
#        print(y_data[y_data.y == 1].shape[0])
 #       print(df.shape[0])
        # -------------------------------
        return x_data, y_data.values.ravel()

    def trainSets(self, x_data, y_data):
        # Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.
        # Use the train_test_split method in sklearn with the parameter 'shuffle' set to true and the 'random_state' set to 614.
        # args: pandas dataframe, pandas dataframe
        # return: pandas dataframe, pandas dataframe, pandas series, pandas series
        # -------------------------------
        # ADD CODE HERE
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.2, shuffle=True, random_state=614)
        # -------------------------------
        return x_train, x_test, y_train, y_test

class NeuralNetwork():

    def dataPreProcess(self, x_train, x_test):
        # Pre-process the data to standardize it, otherwise the grid search will take much longer.
        # args: pandas dataframe, pandas dataframe
        # return: pandas dataframe, pandas dataframe
        # -------------------------------
        # ADD CODE HERE
        scaler = StandardScaler()
        scaler.fit(x_train)
        scaled_x_train = scaler.transform(x_train)
        scaled_x_test = scaler.transform(x_test)
        # -------------------------------
        return scaled_x_train, scaled_x_test


dataset = Data()
nn = NeuralNetwork()
data = 'data/pima-indians-diabetes.csv'
x_data, y_data = dataset.dataAllocation(data)
x_train, x_test, y_train, y_test = dataset.trainSets(x_data, y_data)
x_train_scaled, x_test_scaled = nn.dataPreProcess(x_train, x_test)
schedule = mlrose.ExpDecay(exp_const=0.1)

iterations_dg = [i for i in range(10, 3000, 100)]
iterations_rhc = [i for i in range(10, 3000, 100)]
iterations_sa = [i for i in range(10, 3000, 100)]
iterations_ga = [i for i in range(10, 600, 100)]
accuracy_train_gd = []
accuracy_test_gd = []
accuracy_train_rhc = []
accuracy_test_rhc = []
accuracy_train_sa = []
accuracy_test_sa = []
accuracy_train_ga = []
accuracy_test_ga = []
time_gd = []
time_rhc = []
time_sa = []
time_ga = []
for i in iterations_dg:
    start = time.time()
    nn_model_gd = mlrose.NeuralNetwork(hidden_nodes = [6], activation = 'relu', \
                                 algorithm = 'gradient_descent', max_iters = i, \
                                 bias = True, is_classifier = True, learning_rate = 0.005, \
                                 early_stopping = False, clip_max = 1, max_attempts = 100, \
                                 random_state = 3)
    nn_model_gd.fit(x_train_scaled, y_train)
    end = time.time()
    train_time = end - start
    y_train_pred = nn_model_gd.predict(x_train_scaled)
    y_test_pred = nn_model_gd.predict(x_test_scaled)
    y_train_accuracy = accuracy_score(y_train, y_train_pred)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    accuracy_train_gd.append(y_train_accuracy)
    accuracy_test_gd.append(y_test_accuracy)
    time_gd.append(train_time)

for i in iterations_rhc:
    start = time.time()
    nn_model_rhc = mlrose.NeuralNetwork(hidden_nodes = [6], activation = 'relu', \
                                 algorithm = 'random_hill_climb', max_iters = i, \
                                 bias = True, is_classifier = True, learning_rate = 0.1, \
                                 early_stopping = False, clip_max = 1, max_attempts = 100, \
                                 random_state = 3, pop_size=4000)
    nn_model_rhc.fit(x_train_scaled, y_train)
    end = time.time()
    train_time = end - start
    y_train_pred = nn_model_rhc.predict(x_train_scaled)
    y_test_pred = nn_model_rhc.predict(x_test_scaled)
    y_train_accuracy = accuracy_score(y_train, y_train_pred)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    accuracy_train_rhc.append(y_train_accuracy)
    accuracy_test_rhc.append(y_test_accuracy)
    time_rhc.append(train_time)

for i in iterations_sa:
    start = time.time()
    nn_model_sa = mlrose.NeuralNetwork(hidden_nodes = [6], activation = 'relu', \
                                 algorithm = 'simulated_annealing', schedule = schedule, max_iters = i, \
                                 bias = True, is_classifier = True, learning_rate = 0.5, \
                                 early_stopping = False, clip_max = 1, max_attempts = 100, \
                                 random_state = 3, pop_size=4000)
    nn_model_sa.fit(x_train_scaled, y_train)
    end = time.time()
    train_time = end - start
    y_train_pred = nn_model_sa.predict(x_train_scaled)
    y_test_pred = nn_model_sa.predict(x_test_scaled)
    y_train_accuracy = accuracy_score(y_train, y_train_pred)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    accuracy_train_sa.append(y_train_accuracy)
    accuracy_test_sa.append(y_test_accuracy)
    time_sa.append(train_time)

for i in iterations_ga:
    start = time.time()
    nn_model_ga = mlrose.NeuralNetwork(hidden_nodes = [6], activation = 'relu', \
                                 algorithm = 'genetic_alg', max_iters = i, \
                                 bias = True, is_classifier = True, learning_rate = 0.1, \
                                 early_stopping = False, clip_max = 1, max_attempts = 100, \
                                 random_state = 3, pop_size=2000)
    nn_model_ga.fit(x_train_scaled, y_train)
    end = time.time()
    train_time = end - start
    y_train_pred = nn_model_ga.predict(x_train_scaled)
    y_test_pred = nn_model_ga.predict(x_test_scaled)
    y_train_accuracy = accuracy_score(y_train, y_train_pred)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    accuracy_train_ga.append(y_train_accuracy)
    accuracy_test_ga.append(y_test_accuracy)
    time_ga.append(train_time)

plt.figure(0)
plt.plot(iterations_dg, accuracy_train_gd, label='BP_train')
plt.plot(iterations_dg, accuracy_test_gd, label='BP_test')
plt.plot(iterations_rhc, accuracy_train_rhc, label='RHC_train')
plt.plot(iterations_rhc, accuracy_test_rhc, label='RHC_test')
plt.plot(iterations_sa, accuracy_train_sa, label='SA_train')
plt.plot(iterations_sa, accuracy_test_sa, label='SA_test')
plt.plot(iterations_ga, accuracy_train_ga, label='GA_train')
plt.plot(iterations_ga, accuracy_test_ga, label='GA_test')
plt.xlabel('Number of iterations')
plt.ylabel('Accuracy')
plt.title('Training/Testing accuracy')
plt.legend(loc='lower right')
plt.show()

plt.figure(1)
plt.plot(iterations_dg, time_gd, label='BP_train')
plt.plot(iterations_rhc, time_rhc, label='RHC_train')
plt.plot(iterations_sa, time_sa, label='SA_train')
plt.plot(iterations_ga, time_ga, label='GA_train')
plt.xlabel('Number of iterations')
plt.ylabel('Wall clock time (s)')
plt.title('Computational cost')
plt.legend(loc='lower right')
plt.yscale('log')
plt.show()
