# -*- coding: utf-8 -*-

"""
Created on Wed May 26 09:17:09 2021

@author: Samer Kazem Qarajai

Student ID: 20107283


"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # for plotting
import seaborn as sns
import os
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder

############################## Classification Model ############################


PF_Class = Final_DF.copy()
PF_Class.isna().any()
          
                         

# replacing NA values and renaming some features 

PF_Class['platformType']=PF_Class['platformType'].fillna('No Platform')
PF_Class['country']=PF_Class['country'].fillna('No country')
PF_Class['teamId']=PF_Class['teamId'].fillna(0)
PF_Class['strength']=PF_Class['strength'].fillna(0)
PF_Class['total_purchases_amount']=PF_Class['total_purchases_amount'].fillna(0)
PF_Class['purchases_count']=PF_Class['purchases_count'].fillna(0)
PF_Class['hit_count']=PF_Class['hit_count'].fillna(0)
PF_Class['game_clicks_count']=PF_Class['game_clicks_count'].fillna(0)
PF_Class['ad_clicks_count']=PF_Class['ad_clicks_count'].fillna(0)

PF_Class.isna().any()


# Label Encoding for features : PlatformType and Country
enc = LabelEncoder()

PT_label_encoder = enc.fit(PF_Class['platformType'])
PT_integer_classes = PT_label_encoder.transform(PT_label_encoder.classes_)
t = PT_label_encoder.transform(PF_Class['platformType'])
PF_Class['EncPlatformType'] = t

CO_label_encoder = enc.fit(PF_Class['country'])
CO_integer_classes = CO_label_encoder.transform(CO_label_encoder.classes_)
t = CO_label_encoder.transform(PF_Class['country'])
PF_Class['EncCountry'] = t

# Delete instances where the user has no clicks, no team assigned, no bought items
PF_Class = PF_Class.drop(PF_Class[(PF_Class['teamId'] == 0) & (PF_Class['strength'] == 0) & (PF_Class['game_clicks_count'] == 0) & (PF_Class['purchases_count'] == 0)].index)
       

PF_Class_final = PF_Class.copy()

# Delete Features that are not needed for classfication

del PF_Class_final['platformType']
del PF_Class_final['country']
del PF_Class_final['userId']
del PF_Class['timestamp']

PF_Class_final.isna().any()




X = PF_Class_final.drop('bought_items', axis=1)
Y = PF_Class_final['bought_items']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

clf1 = svm.SVC()
clf1 = clf1.fit(x_train , y_train)


Y_prediction = clf1.predict(x_test)



print(confusion_matrix(y_test, Y_prediction))
print(classification_report(y_test, Y_prediction))
print("Accuracy for SVM : ",accuracy_score(y_test,Y_prediction))




