#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import relplot


def safeSigmoid(x, eps=0):
    y = 1.0/(1.0 + np.exp(-x))
    if eps > 0:
        y[y < eps] = eps
        y[y > 1 - eps] = 1 - eps
    return y

def h(theta, X, eps=0.0):
    return safeSigmoid(X*theta, eps)

def J(h, theta, X, y):
    m = len(y)
    h_val = h(theta, X)
    s1 = np.multiply(y, np.log(h_val))
    s2 = np.multiply((1 - y), np.log(1 - h_val))
    return -np.sum(s1 + s2, axis=0) / m

def dJ(h, theta, X, y):
    return 1.0 / y.shape[0] * (X.T * (h(theta, X) - y))

def GD(h, fJ, fdJ, theta, X, y, alpha=0.01, eps=10**-3, maxSteps=10000):
    errorCurr = fJ(h, theta, X, y)
    errors = [[errorCurr, theta]]
    while True:
        # oblicz nowe theta
        theta = theta - alpha * fdJ(h, theta, X, y)
        # raportuj poziom błędu
        errorCurr, errorPrev = fJ(h, theta, X, y), errorCurr
        # kryteria stopu
        if errorCurr > errorPrev:
            raise Exception('Zbyt duży krok!')
        if abs(errorPrev - errorCurr) <= eps:
            break
        if len(errors) > maxSteps:
            break
        errors.append([errorCurr, theta]) 
    return theta, errors

def classifyBi(h, theta, X):
    probs = h(theta, X)
    result = np.array(probs > 0.5, dtype=int)
    return result, probs

def plot_data_for_classification(X, Y, xlabel, ylabel):    
    fig = plt.figure(figsize=(16*.6, 9*.6))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    X = X.tolist()
    Y = Y.tolist()
    X1n = [x[1] for x, y in zip(X, Y) if y[0] == 0]
    X1p = [x[1] for x, y in zip(X, Y) if y[0] == 1]
    X2n = [x[2] for x, y in zip(X, Y) if y[0] == 0]
    X2p = [x[2] for x, y in zip(X, Y) if y[0] == 1]
    ax.scatter(X1n, X2n, c='r', marker='x', s=50, label='Dane')
    ax.scatter(X1p, X2p, c='g', marker='o', s=50, label='Dane')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.margins(.05, .05)
    return fig

def powerme(x1,x2,n):
    X = []
    for m in range(n+1):
        for i in range(m+1):
            X.append(np.multiply(np.power(x1,i),np.power(x2,(m-i))))
    return np.hstack(X)

def plot_decision_boundary(fig, h, theta, degree):
    ax = fig.axes[0]

    xmin = 1860.0
    xmax = 2020.0
    xstep = 1.0
    xspan = int((xmax - xmin) / xstep)

    ymin = 0.0
    ymax = 1200.0
    ystep = 1.0
    yspan = int((ymax - ymin) / ystep)

    xx, yy = np.meshgrid(np.arange(xmin, xmax, xstep),
                         np.arange(ymin, ymax, ystep))
    l = len(xx.ravel())
    C = powerme(yy.reshape(l, 1), xx.reshape(l, 1), degree)
    z = classifyBi(h, theta, C)[0].reshape(yspan, xspan)

    plt.contour(xx, yy, z, levels=[0.5], colors='m', lw=3)

data = pd.read_csv('wyk/mieszkania4.tsv', sep='\t')
data['Czy kamienica'] = (data['Typ zabudowy'] == 'kamienica')
print(data.columns)

data = data[
    (data['Powierzchnia w m2'] < 10000)
    & (data['cena'] < 10000000)
    ]

relplot(data=data, x='Rok budowy', y='Powierzchnia w m2', hue='Czy kamienica')
plt.show()

# X_columns = ['cena', 'Powierzchnia w m2', 'Rok budowy']
X_columns = ['Rok budowy', 'Powierzchnia w m2']
Y_column = ['Czy kamienica']

data = data[X_columns + Y_column].dropna()

m = len(data)
n = len(X_columns)

X = np.matrix(np.concatenate((np.ones((m, 1)), data[X_columns].values), axis=1))
Y = np.matrix(data[Y_column].values, dtype=int)

split_point = int(0.8 * m)

X_train = X[:split_point]
X_test = X[split_point:]
Y_train = Y[:split_point]
Y_test = Y[split_point:]

thetaStartMx = np.ones((n + 1, 1))
thetaBest, errors = GD(h, J, dJ, thetaStartMx, X_train, Y_train, 
                       alpha=0.1, eps=10**-7, maxSteps=10000)
print(thetaBest)

Y_predicted, Y_probs = classifyBi(h, thetaBest, X_test)

print(Y_predicted.sum())
print(Y_test.sum())

accuracy = np.array(Y_predicted == Y_test, dtype=int).sum() / Y_test.shape[0]
print(accuracy)

fig = plot_data_for_classification(X, Y, xlabel=u'Rok budowy', ylabel=u'Powierzchnia w m2')
plot_decision_boundary(fig, h, thetaBest, 1)
plt.show()


# More dimensions

dim = 2

X_train2 = powerme(X_train[:,1], X_train[:,2], dim)
X_test2 = powerme(X_test[:,1], X_test[:,2], dim)
thetaStart2 = np.ones((X_train2.shape[1], 1))
thetaBest2, errors2 = GD(h, J, dJ, thetaStart2, X_train2, Y_train, 
                         alpha=0.1, eps=10**-7, maxSteps=10000)

print(thetaBest2)
Y_predicted2, Y_probs2 = classifyBi(h, thetaBest2, X_test2)

print(Y_predicted2.sum())
print(Y_test.sum())

accuracy2 = np.array(Y_predicted2 == Y_test, dtype=int).sum() / Y_test.shape[0]
print(accuracy2)

fig2 = plot_data_for_classification(X, Y, xlabel=u'Rok budowy', ylabel=u'Powierzchnia w m2')
plot_decision_boundary(fig2, h, thetaBest2, dim)
plt.show()