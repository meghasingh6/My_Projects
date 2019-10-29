# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:31:37 2019

@author: hp
"""

import pandas as pd
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

pima = pd.read_csv(r"D:\Patients Project\export_dataframe.csv",encoding='latin1')
pima.head()
col_names = ['Lcore', 'Lsurf', 'Lo', 'Lbp', 'Surf', 'ore', 'Bp','Comfort', 'Decision']
feature_cols = ['Bp','Comfort','ore']
C = 1.0 
X = pima[feature_cols] # Features
y = pima.Decision # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33) 
from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear', C=C).fit(X, y)
svclassifier.fit(X_train, y_train)  
y_pred = svclassifier.predict(X_test) 
print("Accuracy with SVM from linear kernel:",metrics.accuracy_score(y_test, y_pred)*100) 
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))   

svclassifie = SVC(kernel='poly', degree=3, C=C).fit(X, y)
svclassifie.fit(X_train, y_train)  
y_pred = svclassifie.predict(X_test) 
print("Accuracy with SVM from polynomial kernel:",metrics.accuracy_score(y_test, y_pred)*100) 
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))   

svclassifi = SVC(kernel='rbf', gamma=0.0001, C=C).fit(X, y)
svclassifi.fit(X_train, y_train)  
y_pred = svclassifi.predict(X_test) 
print("Accuracy with SVM from RBF kernel:",metrics.accuracy_score(y_test, y_pred)*100) 
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))   
