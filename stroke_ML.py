# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 04:05:05 2024

@author: Jevon Putra Joesianto
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Stroke Prediction Data Description
stroke_data = pd.read_csv('stroke_data.csv').dropna()
stroke_data.head()
stroke_data.describe()
stroke_data.columns

# Data Balanced Show
plt.figure()
data_show = stroke_data['stroke'].value_counts()
data_show.plot(kind='bar')
plt.show()
print(data_show)

# Inputs with 10 Features
inputs = ['sex', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status']

# Finding Information Importance Score
# This part is inspired by (Rosidi, 2023)
X = stroke_data[inputs]
Y = stroke_data.stroke
information_gain = mutual_info_regression(X, Y)
print(information_gain)
information_score = {}
for i in range(len(inputs)):
    information_score[inputs[i]] = information_gain[i]
sorted_score = sorted(information_score.items(),key=lambda x: x[1],reverse=True)
score_data = pd.DataFrame.from_dict(sorted_score)
score_data.columns = ['Features','Scores']
score_data = score_data.set_index('Features')
print(score_data)

# Plot bar of Importance Score
plt.figure()
plt.barh(score_data.index,score_data.Scores)
plt.ylabel('Features')
plt.xlabel('Importance Score')
plt.title('Features Importance Score')
plt.show()


# Inputs with 9 Features
inputs_9Features = ['age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status']

# Inputs with 8 features
inputs_8Features = ['age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'avg_glucose_level', 'bmi','smoking_status']


# Showing PCA Component Variance
X = stroke_data[inputs]
pca = PCA().fit(X)
PCA_variance = np.cumsum(pca.explained_variance_ratio_)
print(PCA_variance)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()


def universal_analysis(model_type,inputs):
    ''' Making the analysis for all type of
    machine learning'''
    
    # Train and Test model making
    X = stroke_data[inputs]
    Y = stroke_data.stroke
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    
    # Start the machine learning
    begin = time.time()
    model_type.fit(X_train, Y_train)
    prediction = model_type.predict(X_test)
    prediction_train = model_type.predict(X_train)
    
    # Metrics of machine learning prediction
    accuracy = accuracy_score(Y_test,prediction)
    train_accuracy = accuracy_score(Y_train, prediction_train)
    precision = precision_score(Y_test, prediction)
    recall = recall_score(Y_test,prediction)
    confusion_matrix = metrics.confusion_matrix(Y_test, prediction)
    True_negative = confusion_matrix[0,0]
    False_negative = confusion_matrix[1,0]
    True_positive = confusion_matrix[1,1]
    False_positive = confusion_matrix[0,1]
    
    # 10 Fold Validation
    fold_valid = cross_val_score(model_type, X, Y, cv = 10)
    
    # Print all metric
    print('Train Accuracy:',train_accuracy)
    print('Test Accuracy:',accuracy)
    print('Specificity:', True_negative/(True_negative+False_positive))
    print('Precision:',precision)
    print('Recall:', recall)
    print('----10 Fold Cross Validation----')
    print('Cross Validation Accuracy:',fold_valid.mean())
    print('Cross Validation Variation:',fold_valid.std())
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix).plot();
    print('Time Score:',time.time()-begin)
    
    
def pca_analysis(model_type,inputs):
    ''' Making the analysis for all type of
    machine learning using PCA dimensional reduction'''
    
    # Train and Test model making
    X = stroke_data[inputs]
    Y = stroke_data.stroke
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    
    # Create the PCA
    pca = PCA(n_components=3)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    # Start the machine learning
    begin = time.time()
    model_type.fit(X_train, Y_train)
    prediction = model_type.predict(X_test)
    prediction_train = model_type.predict(X_train)
    
    # Metrics of machine learning prediction
    accuracy = accuracy_score(Y_test,prediction)
    train_accuracy = accuracy_score(Y_train, prediction_train)
    precision = precision_score(Y_test, prediction)
    recall = recall_score(Y_test,prediction)
    confusion_matrix = metrics.confusion_matrix(Y_test, prediction)
    True_negative = confusion_matrix[0,0]
    False_negative = confusion_matrix[1,0]
    True_positive = confusion_matrix[1,1]
    False_positive = confusion_matrix[0,1]
    
    # 10 Fold Validation
    fold_valid = cross_val_score(model_type, X, Y, cv = 10)
    
    # Print all metric
    print('Train Accuracy:',train_accuracy)
    print('Test Accuracy:',accuracy)
    print('Specificity:', True_negative/(True_negative+False_positive))
    print('Precision:',precision)
    print('Recall:', recall)
    print('----10 Fold Cross Validation----')
    print('Cross Validation Accuracy:',fold_valid.mean())
    print('Cross Validation Variation:',fold_valid.std())
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix).plot();
    print('Time Score:',time.time()-begin)


### Random Forest Analysis
rf = RandomForestClassifier(max_depth=13)
rf2 = RandomForestClassifier(max_depth=11)
rf3 = RandomForestClassifier(max_depth=9)
universal_analysis(rf, inputs)
#universal_analysis(rf2, inputs)
#universal_analysis(rf3, inputs)

#universal_analysis(rf, inputs_9Features)
#universal_analysis(rf2, inputs_9Features)
#universal_analysis(rf3, inputs_9Features)

#universal_analysis(rf, inputs_8Features)
#universal_analysis(rf2, inputs_8Features)
#universal_analysis(rf3, inputs_8Features)

#pca_analysis(rf, inputs)
#pca_analysis(rf2, inputs)
#pca_analysis(rf3, inputs)

### Logistic Regression
logicReg = LogisticRegression(C=10)
logicReg2 = LogisticRegression(C=50)
logicReg3 = LogisticRegression(C=100)
universal_analysis(logicReg, inputs)
#universal_analysis(logicReg2, inputs)
#universal_analysis(logicReg3, inputs)

#universal_analysis(logicReg, inputs_9Features)
#universal_analysis(logicReg2, inputs_9Features)
#universal_analysis(logicReg3, inputs_9Features)

#universal_analysis(logicReg, inputs_8Features)
#universal_analysis(logicReg2, inputs_8Features)
#universal_analysis(logicReg3, inputs_8Features)

#pca_analysis(logicReg, inputs)
#pca_analysis(logicReg2, inputs)
#pca_analysis(logicReg3, inputs)

### KNN
'''for i in range(1,11):
    neighbor = KNeighborsClassifier(n_neighbors=i)
    universal_analysis(neighbor, inputs)'''
'''for i in range(1,11):
    neighbor = KNeighborsClassifier(n_neighbors=i)
    universal_analysis(neighbor, inputs_9Features)'''
'''for i in range(1,11):
    neighbor = KNeighborsClassifier(n_neighbors=i)
    universal_analysis(neighbor, inputs_8Features)'''
'''for i in range(1,11):
    neighbor = KNeighborsClassifier(n_neighbors=i)
    pca_analysis(neighbor, inputs)'''

























