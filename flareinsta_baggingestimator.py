from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType
import re, string
import json
import numpy as np
from sklearn import ensemble

from sklearn import metrics
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from collections import Counter

import warnings

warnings.filterwarnings("ignore")

conf = SparkConf().setAppName('Instagram Data Cleaning')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

inputdata = sc.textFile('InstaFlareData.txt')

itr = inputdata.map(lambda line: json.loads(line))

itr = itr.map(lambda l: (int(l.get('ftpcs')), int(l.get('fdur')),  int(l.get('flenergy')), int(l.get('fhenergy')), int(l.get('imonth'))))

itr = itr.map(lambda l: (l[0], l[1], l[2], l[3], l[4]))
test_rdd, train_rdd = itr.randomSplit(weights=[0.2, 0.8], seed=1)

isSchema = StructType([
						StructField('totalpeakcounts', LongType(), False),
                       	StructField('duration', LongType(), False),
                        StructField('flenergy', LongType(), False),
                        StructField('fhenergy', LongType(), False),
                        StructField('monthlabel', IntegerType(), False)])

labeltrain_df = sqlContext.createDataFrame(train_rdd, isSchema)
sqlContext.registerDataFrameAsTable(labeltrain_df, "train_data")

labeltrain_df = labeltrain_df.dropDuplicates()
labeltrain_df = labeltrain_df.dropna()


labeltrain_list = labeltrain_df.collect()
trainlabel = []
for dataitem in labeltrain_list:
    trainlabel.append(dataitem['monthlabel'])

labeltrainwl_df = labeltrain_df['totalpeakcounts', 'duration', 'flenergy', 'fhenergy']
labeltrain_list = labeltrainwl_df.collect()

labeltest_df = sqlContext.createDataFrame(test_rdd, isSchema)
sqlContext.registerDataFrameAsTable(labeltest_df, "test_data")

labeltest_df = labeltest_df.dropDuplicates()
labeltest_df = labeltest_df.dropna()

labeltest_list = labeltest_df.collect()

testlabel = []
for dataitem in labeltest_list:
    testlabel.append(dataitem['monthlabel'])

labeltestwl_df = labeltest_df['totalpeakcounts', 'duration', 'flenergy', 'fhenergy']
labeltest_list = labeltestwl_df.collect()

sampling_rates = [0.2, 0.4, 0.6, 0.8]
forest_sizes = [20, 50, 100, 150]
score = make_scorer(accuracy_score)
parameters = { 'n_estimators' : [20, 50, 100]}


for sampling_rate in sampling_rates:
    print 'sampling rate='+str(sampling_rate)
    accuracy_results = []
    for forest_size in forest_sizes:
        rf_clf = ensemble.BaggingClassifier(max_samples=sampling_rate)
        CV_model = GridSearchCV(rf_clf, parameters, cv=5, scoring = score)
        CV_model.fit(labeltrain_list, trainlabel)
        predictions = CV_model.predict(labeltest_list)
        accuracy = metrics.accuracy_score(testlabel, predictions)
        accuracy_results.append(accuracy)   
        print Counter(predictions)
    print accuracy_results

