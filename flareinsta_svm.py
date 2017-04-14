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

import warnings

warnings.filterwarnings("ignore")

conf = SparkConf().setAppName('Instagram Data Cleaning')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

inputdata = sc.textFile('InstaFlareData.txt')

itr = inputdata.map(lambda line: json.loads(line))

itr = itr.map(lambda l: (int(l.get('ftpcs')), int(l.get('fdur')),  int(l.get('flenergy')), int(l.get('fhenergy')), int(l.get('imonth'))))

itr = itr.map(lambda l: (l[0], pow(l[0],2), l[1], pow(l[1],2),\
						l[2], pow(l[2],2), l[3], pow(l[3],2), l[4]))
test_rdd, train_rdd = itr.randomSplit(weights=[0.1, 0.9], seed=1)

isSchema = StructType([
						StructField('totalpeakcounts', LongType(), False),
                       	StructField('tpcsq', LongType(), False),
                       	StructField('duration', LongType(), False),
                        StructField('dursq', LongType(), False),
                        StructField('flenergy', LongType(), False),
                        StructField('lesq', LongType(), False),
                        StructField('fhenergy', LongType(), False),
                        StructField('hesq', LongType(), False),
                        StructField('monthlabel', IntegerType(), False)])

labeltrain_df = sqlContext.createDataFrame(train_rdd, isSchema)
sqlContext.registerDataFrameAsTable(labeltrain_df, "train_data")

labeltrain_df = labeltrain_df.dropDuplicates()
labeltrain_df = labeltrain_df.dropna()


labeltrain_list = labeltrain_df.collect()
trainlabel = []
for dataitem in labeltrain_list:
    trainlabel.append(dataitem['monthlabel'])

labeltrainwl_df = labeltrain_df['totalpeakcounts', 'tpcsq', 'duration', 'dursq','flenergy', 'lesq', 'fhenergy', 'hesq']
labeltrain_list = labeltrainwl_df.collect()

labeltest_df = sqlContext.createDataFrame(test_rdd, isSchema)
sqlContext.registerDataFrameAsTable(labeltest_df, "test_data")

labeltest_df = labeltest_df.dropDuplicates()
labeltest_df = labeltest_df.dropna()

labeltest_list = labeltest_df.collect()

testlabel = []
for dataitem in labeltest_list:
    testlabel.append(dataitem['monthlabel'])

labeltestwl_df = labeltest_df['totalpeakcounts', 'tpcsq', 'duration', 'dursq','flenergy', 'lesq', 'fhenergy', 'hesq']
labeltest_list = labeltestwl_df.collect()


def sigma_to_gamma(sigma):
    gamma = 1.0 * 1/(2*pow(sigma, 2))
    return gamma

c = [0.001, 0.01, 0.1, 10, 1000, 10000]
gamma = [sigma_to_gamma(1), sigma_to_gamma(100), sigma_to_gamma(10000), sigma_to_gamma(1000000000)]

parameters = {'C':c, 'gamma':gamma}

svr = SVC(kernel='rbf', max_iter=50,  probability = True)
score = make_scorer(accuracy_score)
CV_model = GridSearchCV(svr,parameters, cv=5, scoring = score)
CV_model.fit(labeltrain_list, trainlabel)

prediction = CV_model.predict(labeltrain_list)
val_acc = accuracy_score(trainlabel, prediction)
print ("The Train Accuracy is = " +str(val_acc))

prediction = CV_model.predict(labeltest_list)
val_acc = accuracy_score(testlabel, prediction)
print ("The Validation Accuracy is = " +str(val_acc))
print itr.count()

predictions = CV_model.predict_proba(labeltest_list)
print predictions
