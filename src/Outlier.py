#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 23:26:41 2021

@author: suenchihang
"""

from regression import GenerateData, UniformX
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng()

#generate data with Outliers given orginal X and y
class Outlier(GenerateData): 
    
    def generate_X(self, original_X, original_y, original_beta, positions, magnitude):
        self.X = np.vstack((original_X,np.array((positions))))
        self.N = len(original_X)+len(positions)
        self.X0 = np.ones((self.N,1))
        self.X1 =  np.concatenate([self.X0, self.X], axis=1)
        self.original_X=original_X
        self.original_y=original_y
        self.original_beta=original_beta
        self.positions = positions
        self.magnitude = magnitude
    def generate_y(self):
        self.e_add = self.rng.normal(0, self.e_var**(1/2), size=(len(self.positions),1))
        self.y_add = np.matmul(self.positions, self.original_beta[1:])+ self.e_add + (self.magnitude + self.original_beta[0])
        self.y = np.vstack((self.original_y, self.y_add))
 
#spread positions given start and end points, and handle trivial cases when N=1 or when end-point not given
def Outlier_position(start_position, end_position=[[float("inf"),float("inf")]], N=1):
    if N==1:
        positions = [start_position]
    else:
        if end_position==[[float("inf"),float("inf")]]:
            positions=[start_position for n in range(N)]
        else:
            increment=(np.array(end_position) - np.array(start_position))/(N-1)
            positions=[]
            for n in range(N):
                new = np.add(np.array(start_position) , np.array(n * increment)).tolist()
                positions.append(new)
            
    return positions
    

def print_coef(X, y, to_print=True):
    reg = LinearRegression().fit(X, y)
    score=reg.score(X, y)
    coef=reg.coef_
    intercept=reg.intercept_
    ss_residual = np.linalg.norm(y - reg.predict(X)) ** 2
    if to_print:
        print("intercept: "+str(intercept)+"; coef: "+str(coef))
    return     intercept, coef, score, ss_residual
        



#testing, to be cleared later

test1=UniformX(100)
test1.generate_X()
test1.generate_y()
test1.generate_dataset()
plt.plot(test1.X, test1.y, 'o')
plt.show()

ax = plt.axes(projection='3d')
ax.scatter3D(test1.X[:,0].flatten(), test1.X[:,1].flatten(), test1.y, c=test1.y, cmap='Greens')

#add single outlier and test impact for different positions
set1=UniformX(N=1000)
set1.generate_dataset()
X=set1.X
print(len(X))

y=set1.y
print("set1 - ")
print_coef(set1.X, set1.y)



set1a_X=np.vstack((set1.X,np.array((10,10))))
set1a_y=np.vstack((set1.y, 500))
print(set1.X)
print("set1a -")
print_coef(set1a_X, set1a_y)
print(set1a_X)
print(set1a_y)




set1b_X=np.vstack((set1.X,np.array((0,0))))
set1b_y=np.vstack((set1.y, 500))
print("set1b -")
print_coef(set1b_X, set1b_y)

set1c_X=np.vstack((set1.X,np.array((-10,-10))))
set1c_y=np.vstack((set1.y, 500))
print("set1c -")
print_coef(set1c_X, set1c_y)

#test impact for negative direction
set2a_X=np.vstack((set1.X,np.array((10,10))))
set2a_y=np.vstack((set1.y, -500))
print("set2a -")
print_coef(set2a_X, set2a_y)

set2b_X=np.vstack((set1.X,np.array((0,0))))
set2b_y=np.vstack((set1.y, -500))
print("set2b -")
print_coef(set2b_X, set2b_y)

set2c_X=np.vstack((set1.X,np.array((-10,-10))))
set2c_y=np.vstack((set1.y, -500))
print("set2c -")
print_coef(set2c_X, set2c_y)


test3=Outlier(N=1000)
test3.generate_dataset(original_X=X, original_y=y, original_beta=set1.beta, positions=[[10,10]], magnitude=450)


print("Outlier")
print_coef(test3.X, test3.y)
print(test3.X)
print((test3.y))

print(test3.N)

p=(Outlier_position((10,10),[0,0],N=1))

test4=Outlier(N=1000)
test4.generate_dataset(original_X=X, original_y=y, original_beta=set1.beta, positions=p, magnitude=450)
print("Outlier")
print_coef(test4.X, test4.y)
print(test4.X)
print((test4.y))
print(test4.N)



