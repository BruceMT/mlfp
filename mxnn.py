# coding=utf-8
import numpy as np
import math
import keras as kr
from keras.models import Sequential
from keras.layers import Dense, Activation

def print1d(y):
    x=np.arange(0,y.size,1)
    return x,y


class NN(object):
    model = []

    def __init__(self, layers):
        self.model = Sequential()
        self.model.add(Dense(layers[1][0], activation=layers[1][1], input_dim=layers[0][0]))
        #kr.initializers.RandomUniform(minval=init_weight[0], maxval=init_weight[1], seed=None)
        for i in range(2, len(layers)):
            self.model.add(Dense(layers[i][0], activation=layers[i][1]))

    def fit(self, X, Y, learning_rate=0.1, epochs=2, solver="Adam", momentum=1):
        if solver == "SGD":
            solver = kr.optimizers.SGD(
                lr=learning_rate)
        elif solver == "Adam":
            solver = kr.optimizers.Adam(
                lr=learning_rate)
        self.model.compile(optimizer=solver, loss='binary_crossentropy')  #'mean_squared_error'
        self.model.fit(X, Y, epochs=epochs,verbose=0)

    def predict(self, X):
        X = np.array(X)
        pr = self.model.predict(X)
        return pr

    def loss(self):
        return self.history.losses
