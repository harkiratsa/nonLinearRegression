#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 23:37:45 2018

@author: harkirat
"""


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read dataset
df = pd.read_csv('china_gdp.csv')
year = df[['Year']]
value = df[['Value']]

#converting them into matrix
year = year.values
value = value.values
year = year

#no of data points
m = len(year)
#normalising value by diving with maximum value
max_value = np.max(value)
value = value/max_value

plt.scatter(year,value,color='red')
plt.xlabel('Year')
plt.ylabel('GDP value')
plt.show()

#model selected for purpose is
#y = 1/(1+e^(-beta1*(x-beta2)))
#where y = gdp value and x = year
#since there is no test set for it, we can simply check it by plotting the 
#resultant graph obtained using learned parameters

beta1 = 0.9
beta2 = 2000

def costFunction(year,value,beta1,beta2):
    '''
    costFunction is used for calculating cost/loss related to the model
    It takes year, value matrices along with weights as input
    Returns cost
    '''
    predictions = 1/(1+np.exp(-1*beta1*(year-beta2)))
    cost = -1*(0.5/m)*np.sum(value*np.log(predictions)+(1-value)*np.log(1-predictions))
    return cost

def learningModel(year,value,beta1,beta2,learning_rate=0.003,num_iterations=1000):
    '''
    learningModel is used for learning required parameters for the proper functioning of model
    It takes year and value matrices along with weights, learning_rate and num_iterations as parameters
    Returns learned parameters
    '''
    cost = []
    for i in range(num_iterations):
        
        predictions = 1/(1+np.exp(-1*beta1*(year-beta2)))
        dJdy = -1*(value/predictions - (1-value)/(1-predictions))
        intermediate = (np.exp(-beta1*(year-beta2)))/((1+np.exp(-beta1*(year-beta2)))**2)
        dydb1 = intermediate*(year-beta2)
        dydb2 = -1*intermediate*beta1
        dbeta1 = np.sum(dJdy*dydb1)
        dbeta2 = np.sum(dJdy*dydb2)
        beta1 = beta1 - learning_rate*dbeta1
        beta2 = beta2 - learning_rate*dbeta2
        cost.append(costFunction(year,value,beta1,beta2))
    plt.plot(range(num_iterations),cost)
        
    return (beta1,beta2)

cost = costFunction(year,value,beta1,beta2)
print('Initial cost',cost)

parameters = learningModel(year,value,beta1,beta2,num_iterations=50000)
print(parameters)

beta1 = parameters[0]
beta2 = parameters[1]
predictions = 1/(1+np.exp(-1*beta1*(year-beta2)))
plt.close()
plt.scatter(year,value,color='red')
plt.scatter(year,predictions,color='blue')
plt.show()
cost = costFunction(year,value,beta1,beta2)
print('Final Cost',cost)

    
    
