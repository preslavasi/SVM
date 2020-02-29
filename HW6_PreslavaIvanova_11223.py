# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 14:59:30 2018

@author: User
"""
#Importing data
import numpy as np
from numpy import loadtxt
from sklearn import svm
import matplotlib.pyplot as plt
data = loadtxt('cpuLRN0.txt', skiprows=1, delimiter=',')
X = data[:, :2]
y = data[:, 2]

#Applying the SVM function
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
 np.arange(y_min, y_max, h))
X_plot = np.c_[xx.ravel(), yy.ravel()]

#Creating the model
C = 175
svc = svm.SVC(kernel='linear', C=C, decision_function_shape='ovr').fit(X, y)
z = svc.predict(X_plot)
z = z.reshape(xx.shape)

#Showing results on a plot
plt.figure(figsize=(5, 5))
plt.contourf(xx, yy, z, cmap=plt.cm.viridis, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.tab20b)
plt.xlabel('X values')
plt.ylabel('y values')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()
print 'SVM.score for svc :',svc.score(X, y) 

#Creating the  model
c = 1000
svc1 = svm.SVC(kernel='rbf', C=c, decision_function_shape='ovr').fit(X, y)
z1 = svc1.predict(X_plot)
z1 = z1.reshape(xx.shape)

#Showing results on a plot
plt.figure(figsize=(5, 5))
plt.contourf(xx, yy, z1, cmap=plt.cm.plasma, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
plt.xlabel('X values')
plt.ylabel('y values')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with RBF kernel')
plt.show()
print 'SVM.score for svc1 :',svc1.score(X, y)

#Making prediction for point A
sl=6.1
sw=37.05
plt.figure()
plt.pcolormesh(xx, yy, z1, cmap=plt.cm.RdPu)

#Plotting the points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.summer)
plt.scatter(sl,sw, marker='x', color='black')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('3-Class classification from SVM')
plt.show()

#Defining class
dataClass = svc.predict([[sl,sw]])
print('Predicted class: '),
if dataClass == 0:
    print('1')
elif dataClass == 1:
    print('2')
else:
    print('3')
   
#Making prediction for point B    
sl=6.9
sw=32.9
plt.figure()
plt.pcolormesh(xx, yy, z, cmap=plt.cm.RdPu)

#Plotting the points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.summer)
plt.scatter(sl,sw, marker='x', color='black')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('3-Class classification from SVM')
plt.show()

#Defining class
dataClass = svc1.predict([[sl,sw]])
print('Predicted class: '),
if dataClass == 0:
    print('1')
elif dataClass == 1:
    print('2')
else:
    print('3')