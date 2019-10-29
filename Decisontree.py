# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 13:40:44 2019

@author: hp
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
col_names = ['Lcore', 'Lsurf', 'Lo', 'Lbp', 'Surf', 'ore', 'Bp','Comfort', 'Decision']
# load dataset
pima = pd.read_csv(r"D:\Patients Project\export_dataframe.csv",encoding='latin1')
pima.head()
feature_cols = ['Bp','Comfort','ore']
X = pima[feature_cols] # Features
y = pima.Decision # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,stratify =y)
#DECISION TREE CLASSIFIER
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy with decsion tree classifier:",metrics.accuracy_score(y_test, y_pred))
# Accuracy with entropy as criteria
cl = DecisionTreeClassifier(criterion = "entropy", 
			max_depth = 5)
cl = cl.fit(X_train,y_train)
y_pred = cl.predict(X_test)
print("Accuracy with entropy as criteria:",metrics.accuracy_score(y_test, y_pred))
#Accuracy with ginni as criteria
l = DecisionTreeClassifier(criterion = "gini", 
			max_depth=3)
l = cl.fit(X_train,y_train)
y_pred = l.predict(X_test)
print("Accuracy with ginni as criteria:",metrics.accuracy_score(y_test, y_pred))
# =============================================================================
# #Import svm model
# from sklearn import svm
# 
# #Create a svm Classifier
# fd = svm.SVC(kernel='linear') # Linear Kernel
# 
# #Train the model using the training sets
# fd.fit(X_train, y_train)
# 
# #Predict the response for test dataset
# y_pred = fd.predict(X_test)
# 
# 
# #Import scikit-learn metrics module for accuracy calculation
# from sklearn import metrics
# 
# # Model Accuracy: how often is the classifier correct?
# print("Accuracy with svm classifier:",metrics.accuracy_score(y_test, y_pred))
# 
# #Import knearest neighbors Classifier model
# from sklearn.neighbors import KNeighborsClassifier
# 
# #Create KNN Classifier
# knn = KNeighborsClassifier(n_neighbors=7)
# 
# #Train the model using the training sets
# knn.fit(X_train, y_train)
# 
# #Predict the response for test dataset
# y_pred = knn.predict(X_test)
# #Import scikit-learn metrics module for accuracy calculation
# from sklearn import metrics
# # Model Accuracy, how often is the classifier correct?
# print("Accuracy with knn:",metrics.accuracy_score(y_test, y_pred))
# 
# #Import Gaussian Naive Bayes model
# from sklearn.naive_bayes import GaussianNB
# 
# #Create a Gaussian Classifier
# gnb = GaussianNB()
# 
# #Train the model using the training sets
# gnb.fit(X_train, y_train)
# 
# #Predict the response for test dataset
# y_pred = gnb.predict(X_test)
# from sklearn import metrics
# 
# # Model Accuracy, how often is the classifier correct?
# print("Accuracy with naive bayes:",metrics.accuracy_score(y_test, y_pred))
# 
# 
# 
# 
# =============================================================================
