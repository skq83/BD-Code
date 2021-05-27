# -*- coding: utf-8 -*-

"""
Created on Wed May 26 09:17:09 2021

@author: Samer Kazem Qarajai

Student ID: 20107283


"""

import pandas as pd
import numpy as np
from numpy import array
import sys
import os
import findspark
import pyspark
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pandas import DataFrame

############################## KMean Clustering Model ############################

os. getcwd()
os.chdir('D:\BCU University\Course\CMP7203-A-S2-20201 - Dr Besher Alhalabi\Assessment\Final_Code')


findspark.init('C:\spark')
spark = SparkSession.builder \
           .master('local[*]') \
           .appName('My PF App') \
           .getOrCreate()
           
           
trainingDF = PF_Class.copy()           
trainingDF.isna().any()

trainingDF_Final = trainingDF[['total_purchases_amount','strength']]

pDF = spark.createDataFrame(trainingDF_Final)
pDF.printSchema()


#Train KMeans model to create two clusters


parsedData = pDF.rdd.map(lambda line: array([line[0], line[1]])) 


from math import sqrt

def error(point): 
    center = clusters.centers[clusters.predict(point)] 
    return sqrt(sum([x**2 for x in (point - center)]))

kval=[]
wssse=[]
for k in range(2,10):
    clusters = KMeans.train(parsedData, k=k, maxIterations=10,runs=10, initializationMode="random")
    wssse.append(parsedData.map(lambda point:error(point)).reduce(lambda x, y: x+y))
    kval.append(k)


kval_df = DataFrame(kval,columns=['k'])
wssse_df = DataFrame(wssse,columns=['wssse'])

plt.plot(kval_df, wssse_df)
plt.show()


clusters_Final = KMeans.train(parsedData, 5, maxIterations=10, initializationMode="random")


WSSSE_Final = (parsedData.map(lambda point:error(point)).reduce(lambda x, y: x+y))
print('k= ' + str(5) + ' WSSSE = ' + str(WSSSE_Final))



#Display the centers of two clusters
print(clusters_Final.centers)
CCenters=clusters_Final.centers

