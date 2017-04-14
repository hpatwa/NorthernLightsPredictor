# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 2017

@author: HP41293
"""

import xlrd
import datetime
import sys

from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

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
                        StructField('label', DoubleType(), True)])

def udf_updatinglabels(data):
    if(data['lpeakenergy'] < 6 and data['hpeakenergy'] < 12):
        data['label'] = 0.0
    else:
        if(data['lpeakenergy'] >= 25 and data['hpeakenergy'] >= 50):
            data['label'] = 1.0
        else:
            if(data['peakcpers'] >= 100 and data['totalpeakcnts'] >= 30000):
                data['label'] = 1.0
            else:
                data['label'] = 0.0
    
    
for dataitem in finalData:
    udf_updatinglabels(dataitem)

finalDataRDD = sc.parallelize(finalData)    
test_rdd, train_rdd = finalDataRDD.randomSplit(weights=[0.2, 0.8], seed=1)


labeltrain_df = sqlContext.createDataFrame(train_rdd, isSchema)
sqlContext.registerDataFrameAsTable(labeltrain_df, "train_data")
labeltrain_df.show()

unlabeltest_df = sqlContext.createDataFrame(test_rdd, isSchema)
sqlContext.registerDataFrameAsTable(unlabeltest_df, "test_data")

#Begnning the model
assembler = VectorAssembler(
    inputCols=["flareid", "dursec", "peakcpers", "totalpeakcnts", "lpeakenergy", "hpeakenergy"],
    outputCol="features")

lrmodel = LogisticRegression(maxIter = 10, labelCol="label", featuresCol="features")

pipeline=Pipeline(stages=[assembler, lrmodel])

parameter_Grid = ParamGridBuilder() \
	    .addGrid(lrmodel.regParam, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) \
	    .build()

cross_validation = CrossValidator(estimator = pipeline,
                          estimatorParamMaps = parameter_Grid,
                          evaluator = BinaryClassificationMetrics(),
                          numFolds = 5)

crossval_Model = cross_validation.fit(labeltrain_df)
predictions = crossval_Model.transform(labeltrain_df)
trainevaluator = BinaryClassificationMetrics(metricName = "rmse", labelCol = "label", predictionCol = "prediction")
trainRMSE = trainevaluator.evaluate(predictions)

predictions = crossval_Model.transform(unlabeltest_df)
evaluator = RegressionEvaluator(metricName = "rmse", labelCol = "label", predictionCol = "prediction")
testRMSE = evaluator.evaluate(predictions)

print ("RMSE Train = " + str(trainRMSE))
print("RMSE Validation = " + str(testRMSE))