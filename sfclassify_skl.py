# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:33:11 2017

@author: HP41293
"""

import xlrd
import datetime
import sys

from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

conf = SparkConf().setAppName('PBD2 Project')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)


sf_workbook = xlrd.open_workbook('Solar_Flare_Data3.xlsx')
xl_sheet = sf_workbook.sheet_by_index(0)

finalData = []
for row_index in range(1, xl_sheet.nrows):
    row = xl_sheet.row(row_index)
    flareid = int(xl_sheet.cell_value(rowx=row_index, colx=0))
    dursec = int(xl_sheet.cell_value(rowx=row_index, colx=5))
    peakcpers = int(xl_sheet.cell_value(rowx=row_index, colx=6))
    totalpeakcnts = int(xl_sheet.cell_value(rowx=row_index, colx=7))
    lpeakenergy = int(xl_sheet.cell_value(rowx=row_index, colx=18))
    hpeakenergy = int(xl_sheet.cell_value(rowx=row_index, colx=19))
    label = str('null')
    finalData.append({'flareid': flareid, 'dursec': dursec, 'peakcpers': peakcpers,\
                      'totalpeakcnts': totalpeakcnts, 'lpeakenergy': lpeakenergy, \
                      'hpeakenergy': hpeakenergy, 'label': label})

isSchema = StructType([
                       StructField('flareid', IntegerType(), False),
                        StructField('dursec', IntegerType(), False),
                        StructField('peakcpers', IntegerType(), False),
                        StructField('totalpeakcnts', IntegerType(), False),
                        StructField('lpeakenergy', IntegerType(), False),
                        StructField('hpeakenergy', IntegerType(), False),
                        StructField('label', IntegerType(), True)])

def udf_updatinglabels(data):
    if(data['lpeakenergy'] < 6 and data['hpeakenergy'] < 12):
        data['label'] = 0
    else:
        if(data['lpeakenergy'] >= 25 and data['hpeakenergy'] >= 50):
            data['label'] = 1
        else:
            if(data['peakcpers'] >= 100 and data['totalpeakcnts'] >= 30000):
                data['label'] = 1
            else:
                data['label'] = 0
    
    
for dataitem in finalData:
    udf_updatinglabels(dataitem)


finalDataRDD = sc.parallelize(finalData)    
test_rdd, train_rdd = finalDataRDD.randomSplit(weights=[0.2, 0.8], seed=1)

labeltrain_df = sqlContext.createDataFrame(train_rdd, isSchema)
sqlContext.registerDataFrameAsTable(labeltrain_df, "train_data")

labeltrain_list = labeltrain_df.collect()
trainlabel = []
for dataitem in labeltrain_list:
    trainlabel.append(dataitem['label'])

labeltrainwl_df = labeltrain_df['flareid', 'dursec', 'peakcpers', 'totalpeakcnts', 'lpeakenergy', 'hpeakenergy']
labeltrain_list = labeltrainwl_df.collect()

unlabeltest_df = sqlContext.createDataFrame(test_rdd, isSchema)
sqlContext.registerDataFrameAsTable(unlabeltest_df, "test_data")

labeltest_list = unlabeltest_df.collect()
testlabel = []
for dataitem in labeltest_list:
    testlabel.append(dataitem['label'])

unlabeltestwl_df = unlabeltest_df['flareid', 'dursec', 'peakcpers', 'totalpeakcnts', 'lpeakenergy', 'hpeakenergy']
unlabeltest_list = unlabeltestwl_df.collect()


model = LogisticRegression(max_iter= 10)
model.fit(labeltrain_list,trainlabel)

predictedLabelsTrain = model.predict(labeltrain_list)
accuracytrain = metrics.accuracy_score(trainlabel, predictedLabelsTrain)

print "Train Data Accuracy = " + str(accuracytrain)

predictedLabelsTest = model.predict(unlabeltest_list)
accuracytest = metrics.accuracy_score(testlabel, predictedLabelsTest)

print "Test Data Accuracy = " + str(accuracytest)

trainCompleteData = []
i = 0
for item in labeltrain_list:
	flareid = item['flareid']
	dursec = item['dursec']
	peakcpers = item['peakcpers']
	totalpeakcnts = item['totalpeakcnts']
	lpeakenergy = item['lpeakenergy']
	hpeakenergy = item['hpeakenergy']
	label = trainlabel[i]
	trainCompleteData.append({'flareid': flareid, 'dursec': dursec, 'peakcpers': peakcpers,\
					'totalpeakcnts': totalpeakcnts, 'lpeakenergy': lpeakenergy, \
					'hpeakenergy': hpeakenergy, 'label': label})
	i = i+1
TrainCmpltDataRDD = sc.parallelize(trainCompleteData)
tcdf = sqlContext.createDataFrame(TrainCmpltDataRDD, isSchema)
sqlContext.registerDataFrameAsTable(tcdf, "train_data")

testCompleteData = []
i = 0
for item in unlabeltest_list:
	flareid = item['flareid']
	dursec = item['dursec']
	peakcpers = item['peakcpers']
	totalpeakcnts = item['totalpeakcnts']
	lpeakenergy = item['lpeakenergy']
	hpeakenergy = item['hpeakenergy']
	label = testlabel[i]
	testCompleteData.append({'flareid': flareid, 'dursec': dursec, 'peakcpers': peakcpers,\
					'totalpeakcnts': totalpeakcnts, 'lpeakenergy': lpeakenergy, \
					'hpeakenergy': hpeakenergy, 'label': label})
	i = i+1
TestCmpltDataRDD = sc.parallelize(testCompleteData)
tscdf = sqlContext.createDataFrame(TestCmpltDataRDD, isSchema)
sqlContext.registerDataFrameAsTable(tscdf, "test_data")

cmpltdf = tcdf.unionAll(tscdf)

cmpltdf.toPandas().to_csv('CompleteLabeledData.csv')