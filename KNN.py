# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:00:34 2019

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
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=7)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))