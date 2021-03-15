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
from sklearn.metrics import confusion_matrix


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
schedule = mlrose.ExpDecay(exp_const=0.00001)

gd1 = mlrose.NeuralNetwork(hidden_nodes = [6], activation = 'relu', \
                                 algorithm = 'gradient_descent', max_iters = 200, \
                                 bias = True, is_classifier = True, learning_rate = 0.005, \
                                 early_stopping = False, clip_max = 1, max_attempts = 100, \
                                 random_state = 3, curve=True)
gd1_fit = gd1.fit(x_train_scaled, y_train)
gd1_pred = gd1.predict(x_train_scaled)

cm = confusion_matrix(y_train, gd1_pred, normalize='true')
print(cm)

gd11_pred = gd1.predict(x_test_scaled)
cm1 = confusion_matrix(y_test, gd11_pred, normalize='true')
print(cm1)

gd2 = mlrose.NeuralNetwork(hidden_nodes = [6], activation = 'relu', \
                                 algorithm = 'gradient_descent', max_iters = 200, \
                                 bias = True, is_classifier = True, learning_rate = 0.01, \
                                 early_stopping = False, clip_max = 1, max_attempts = 100, \
                                 random_state = 3, curve=True)
gd2_fit = gd2.fit(x_train_scaled, y_train)
gd2pred = gd2.predict(x_train_scaled)

gd3 = mlrose.NeuralNetwork(hidden_nodes = [6], activation = 'relu', \
                                 algorithm = 'gradient_descent', max_iters = 200, \
                                 bias = True, is_classifier = True, learning_rate = 0.001, \
                                 early_stopping = False, clip_max = 1, max_attempts = 100, \
                                 random_state = 3, curve=True)
gd3_fit = gd3.fit(x_train_scaled, y_train)
gd3pred = gd3.predict(x_train_scaled)

plt.plot(gd1.fitness_curve,label='BP_lr0005')
plt.plot(gd2.fitness_curve,label='BP_lr001')
plt.plot(gd3.fitness_curve,label='BP_lr0001')
plt.xlabel('Number of iterations')
plt.ylabel('Fitness')
plt.title('Convergence of BP')
plt.legend(loc='lower right')
plt.show()

