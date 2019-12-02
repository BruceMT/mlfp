
# coding=utf-8
import numpy as np
import math
import keras as kr
from keras.models import Sequential
from keras.layers import Dense


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x) * np.tanh(x)


def logistics(x):
    return 1 / (1+np.exp(-x))


def logistics_deriv(x):
    return logistics(x)*(1-logistics(x))


def sigmoid(x):
    return 1/(1+math.exp(-x))


def sigmoid_deriv(x):
    return sigmoid(x)*(1-sigmoid(x))


class NN:

    activation = tanh
    activation_deriv = tanh_deriv
    weight = []
    bp = []  # 存偏导数
    state = []
    dw = []
    model = []

    def __init__(self, layers, activation='sigmoid', ita=1):
        if activation == 'logistic':
            self.activation = logistics
            self.activation_deriv = logistics_deriv
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_deriv = sigmoid_deriv
        self.weight = []
        self.model = Sequential()

        for i in range(1, len(layers)):
            self.weight.append(
                (2*np.random.random((layers[i], layers[i-1]+1))-1)*ita)
            if i == 1:
                self.model.add(
                    Dense(units=layers[i], activation=activation, input_dim=layers[0]))
            else:
                self.model.add(Dense(units=layers[i], activation=activation))
            self.bp.append([1]*(layers[i]))

    def fit(self, X, Y, learning_rate=0.2, epochs=10000, solver="SGD", momentum=1,loss_func='categorical_crossentropy', batch_size=32):
        X = np.array(X)
        Y = np.array(Y)
        dw = np.array(self.weight).copy()
        if solver=="SGD":
            solver=kr.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        elif solver == "Adam":
            solver=kr.optimizers.Adam(learning_rate=learning_rate,momentum=momentum)
        self.model.compile(loss=loss_func,
                           optimizer=solver,
                           metrics=['accuracy'])
        self.model.fit(X, X, epochs=epochs, batch_size=batch_size)
        #self nerual network forward&&back
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i].tolist()]
            for l in range(len(self.bp)):
                a.append(self.bp[l])
            # fw
            self.state = a
            for l in range(len(self.weight)):
                self.state[l] = self.state[l]+[1]
                for c in range(len(self.weight[l])):
                    j = np.dot(self.state[l], self.weight[l][c])
                    self.state[l+1][c] = self.activation(j)
            rs = Y[i]
            if math.isnan(rs):
                rs = 0
            dw = []
            me = []
            l = len(self.state)

            
            for l in range(l-1, 0, -1):
                if l == len(self.state)-1:
                    for c in range(len(self.state[l])):
                        self.bp[l-1][c] = -(rs-self.state[l][c]) * \
                            (1-self.state[l][c])*self.state[l][c]
                        for i in range(len(a[l-1])):
                            dw = dw+[self.bp[l-1][c] *
                                        a[l-1][i]*learning_rate]
                        self.weight[l-1][c] = self.weight[l-1][c]-dw
                else:
                    # dio[0i] = sum dio[1j] * w[ij] * o[1j]（1-o[1j]）
                    # dw[ij](N)=ita * dio[0i] * o[-1] + alpha * dw[ij](N-1)
                    for c in range(len(self.state[l])-1):
                        self.bp[l-1][c] = 0
                        ee = 1-self.state[l][c]
                        ee = ee*self.state[l][c]
                        for j in range(len(self.bp[l])):
                            eeee = ee*self.bp[l][j]
                            eeee = eeee*self.weight[l][j][c]
                            self.bp[l-1][c] = self.bp[l-1][c]+eeee
                        ff = np.array(self.state[l-1])*self.bp[l-1][c]
                        dw = np.array(a[l-1])*self.bp[l-1][c]*learning_rate+ff
                        self.weight[l-1][c] = self.weight[l -1][c]-dw*learning_rate

    def predict(self, X,y):
        X = np.array(X)
        a = [X.tolist()]
        for l in range(len(self.bp)):
            a.append(self.bp[l])
        # fw
        self.state = a
        for l in range(len(self.weight)):
            self.state[l] = self.state[l]+[1]
            for c in range(len(self.weight[l])):
                j = np.dot(self.state[l], self.weight[l][c])
                self.state[l+1][c] = self.activation(j)
        self.state[-1] = self.model.predict(X)
        if len(y)>0:
            lo = model.evaluate(X, y)
            return self.state[-1], lo
        return self.state[-1]

# test

"""
nn = NN([2, 4, 1])

aa = [[1, 0], [0, 1]]
dd = [1, 0]

nn.fit(aa, dd, learning_rate=0.3)

cc = nn.predict([1, 0])
print(cc)
"""