# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 18:23:31 2019

@author: hp
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
col_names = ['Cement','Slag','Fly ash','Water',
             'SP','Coarse Aggr.','Fine Aggr.',
             'SLUMP','FLOW','CompressiveStrength']
# load dataset
pima = pd.read_csv(r"D:\Slump Project\slump_test.csv",encoding='latin1')
print(pima.head())
pima.info()
#print (pima.DESCR)
df = DataFrame(pima,columns=['Cement','Slag','Fly ash','Water',
             'SP','Coarse Aggr.','Fine Aggr.',
             'SLUMP','FLOW','CompressiveStrength'])
import pylab as pl
# =============================================================================
# pl.suptitle("Histogram for each numeric input variable")
# plt.savefig('fruits_hist')
# plt.show()
# plt.scatter(df['Cement'], df['CompressiveStrength'], color='red')
# plt.title('CompressiveStrength Vs Cement', fontsize=14)
# plt.xlabel('Cement', fontsize=14)
# plt.ylabel('CompressiveStrength', fontsize=14)
# plt.grid(True)
# plt.show()
#  
# plt.scatter(df['Slag'], df['CompressiveStrength'], color='green')
# plt.title('CompressiveStrength Vs Slag', fontsize=14)
# plt.xlabel('Slag', fontsize=14)
# plt.ylabel('CompressiveStrength', fontsize=14)
# plt.grid(True)
# plt.show()
# 
# plt.scatter(df['Fly ash'], df['CompressiveStrength'], color='green')
# plt.title('CompressiveStrength Vs Fly ash', fontsize=14)
# plt.xlabel('Fly ash', fontsize=14)
# plt.ylabel('CompressiveStrength', fontsize=14)
# plt.grid(True)
# plt.show()
# 
# plt.scatter(df['Water'], df['CompressiveStrength'], color='green')
# plt.title('CompressiveStrength Vs Water', fontsize=14)
# plt.xlabel('Water', fontsize=14)
# plt.ylabel('CompressiveStrength', fontsize=14)
# plt.grid(True)
# plt.show()
# 
# plt.scatter(df['SP'], df['CompressiveStrength'], color='green')
# plt.title('CompressiveStrength Vs SP', fontsize=14)
# plt.xlabel('SP', fontsize=14)
# plt.ylabel('CompressiveStrength', fontsize=14)
# plt.grid(True)
# plt.show()
# 
# plt.scatter(df['Coarse Aggr.'], df['CompressiveStrength'], color='green')
# plt.title('CompressiveStrength Vs Coarse Aggr.', fontsize=14)
# plt.xlabel('Coarse Aggr.', fontsize=14)
# plt.ylabel('CompressiveStrength', fontsize=14)
# plt.grid(True)
# plt.show()
# 
# plt.scatter(df['Fine Aggr.'], df['CompressiveStrength'], color='green')
# plt.title('CompressiveStrength Vs Fine Aggr.', fontsize=14)
# plt.xlabel('Fine Aggr.', fontsize=14)
# plt.ylabel('CompressiveStrength', fontsize=14)
# plt.grid(True)
# plt.show()
# 
# plt.scatter(df['SLUMP'], df['CompressiveStrength'], color='green')
# plt.title('CompressiveStrength Vs SLUMP', fontsize=14)
# plt.xlabel('SLUMP', fontsize=14)
# plt.ylabel('CompressiveStrength', fontsize=14)
# plt.grid(True)
# plt.show()
# 
# plt.scatter(df['FLOW'], df['CompressiveStrength'], color='green')
# plt.title('CompressiveStrength Vs FLOW', fontsize=14)
# plt.xlabel('FLOW', fontsize=14)
# plt.ylabel('CompressiveStrength', fontsize=14)
# plt.grid(True)
# plt.show()
# =============================================================================
from sklearn import linear_model
import statsmodels.api as sm
X = df[['Cement','Slag','Fly ash','Water',
             'SP','Coarse Aggr.','Fine Aggr.',
             'SLUMP','FLOW']]
y = df['CompressiveStrength']
 
# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
print('')
print(list(zip(X, regr.coef_)))
# =============================================================================
lm1 = sm.OLS(y,X).fit()
predictions = lm1.predict(X)
print('')
print(lm1.summary())
predictions = regr.predict(X)
print('')
print(predictions[0:5])
print('')
print(regr.score(X,y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
y_pred = regr.predict(X_test)
print('')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# =============================================================================
# plt.scatter(X_test, y_test,  color='gray')
# plt.plot(X_test, y_pred, color='red', linewidth=2)
# plt.show()
# =============================================================================
#GUI
import tkinter as tk 
import statsmodels.api as sm

X = df[['Cement','Fly ash']] # here we have 2 input variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
y = df['CompressiveStrength'] # output variable (what we are trying to predict)
# =============================================================================
# print('Mean Absolute Error:', metrics.mean_absolute_error(y1_test, y1_pred))  
# print('Mean Squared Error:', metrics.mean_squared_error(y1_test, y1_pred))  
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y1_test, y1_pred)))
# 
# =============================================================================
# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, y)
print('')
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
print('')
New_SP= 9
New_SLUMP = 23
print ('Predicted CompressiveStrength \n', regr.predict([[New_SP ,New_SLUMP]]))
print('')
# tkinter GUI
root= tk.Tk() 
 
canvas1 = tk.Canvas(root, width = 1200, height = 450)
canvas1.pack()

# with sklearn
Intercept_result = ('Intercept: ', regr.intercept_)
label_Intercept = tk.Label(root, text=Intercept_result, justify = 'center')
canvas1.create_window(260, 220, window=label_Intercept)

# with sklearn
Coefficients_result  = ('Coefficients: ', regr.coef_)
label_Coefficients = tk.Label(root, text=Coefficients_result, justify = 'center')
canvas1.create_window(260, 240, window=label_Coefficients)

# with statsmodels
print_model = lm1.summary()
label_model = tk.Label(root, text=print_model, justify = 'center', relief = 'solid', bg='LightSkyBlue1')
canvas1.create_window(800, 220, window=label_model)


# New_Interest_Rate label and input box
label1 = tk.Label(root, text='Type Cement: ')
canvas1.create_window(100, 100, window=label1)

entry1 = tk.Entry (root) # create 1st entry box
canvas1.create_window(270, 100, window=entry1)

# New_Unemployment_Rate label and input box
label2 = tk.Label(root, text=' Type Fly ash: ')
canvas1.create_window(120, 120, window=label2)

entry2 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 120, window=entry2)


def values(): 
    global New_Interest_Rate #our 1st input variable
    New_Interest_Rate = float(entry1.get()) 
    
    global New_Unemployment_Rate #our 2nd input variable
    New_Unemployment_Rate = float(entry2.get()) 
    
    Prediction_result  = ('Predicted CompressiveStrength: ', regr.predict([[New_Interest_Rate ,New_Unemployment_Rate]]))
    label_Prediction = tk.Label(root, text= Prediction_result, bg='orange')
    canvas1.create_window(260, 280, window=label_Prediction)
    
button1 = tk.Button (root, text='Predict CompressiveStrength',command=values, bg='orange') # button to call the 'values' command above 
canvas1.create_window(270, 150, window=button1)
 

root.mainloop()
