from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import re, string
import json
import xlrd
import datetime
 

conf = SparkConf().setAppName('Instagram Data Cleaning')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

instagraminput = sc.textFile('InstagramFinalData.txt')
itr = instagraminput.map(lambda line: json.loads(line))

data = itr.map(lambda l: (l.get("urllink"), l.get("isvideo"), (l.get("metadata")).split()))
data = data.map(lambda m: (m[0], m[1], m[2][len(m[2])-6], (m[2][len(m[2])-5]).strip().split(','), int(m[2][len(m[2])-4])))

def change_month_to_no(mnth):
	if(mnth=='Jan'):
		monthno = 1
	else:
		if(mnth=='Feb'):
			monthno = 2
		else:
			if(mnth=='Mar'):
				monthno = 3
			else:
				if(mnth=='Apr'):
					monthno = 4
				else:
					if(mnth=='May'):
						monthno = 5
					else:
						if(mnth=='Jun'):
							monthno = 6
						else:
							if(mnth=='Jul'):
								monthno = 7
							else:
								if(mnth=='Aug'):
									monthno = 8
								else:
									if(mnth=='Sep'):
										monthno = 9
									else:
										if(mnth=='Oct'):
											monthno = 10
										else:
											if(mnth=='Nov'):
												monthno = 11
											else:
												if(mnth=='Dec'):
													monthno = 12
	return monthno

data = data.map(lambda m: (m[0], m[1], change_month_to_no(m[2]), int(m[3][0]), m[4]))
datalist = data.collect()
instadata = []
i = 0
for item in datalist:
	i = i+1
	index = i
	urllink = item[0]
	isVideo = item[1]
	month = item[2]
	date = item[3]
	year = item[4]
	instadata.append({'index': index, 'urllink': urllink, 'isVideo': isVideo, 'month': month, 'date': date, 'year': year})

data = sc.parallelize(instadata)

isSchema = StructType([
                       StructField('index', IntegerType(), False),
                       StructField('urllink', StringType(), False),
                        StructField('isVideo', IntegerType(), False),
                        StructField('month', IntegerType(), False),
                        StructField('date', IntegerType(), False),
                        StructField('year', IntegerType(), False)])


instadata_df = sqlContext.createDataFrame(data, isSchema)
sqlContext.registerDataFrameAsTable(instadata_df, "instagram_data")


################################################################################################

sf_workbook = xlrd.open_workbook('Data.xlsx')
xl_sheet = sf_workbook.sheet_by_index(0)

finalData = []
for row_index in range(1, xl_sheet.nrows):
    row = xl_sheet.row(row_index)
    flareid = int(xl_sheet.cell_value(rowx=row_index, colx=0))
    dursec = int(xl_sheet.cell_value(rowx=row_index, colx=1))
    peakcpers = int(xl_sheet.cell_value(rowx=row_index, colx=2))
    totalpeakcnts = int(xl_sheet.cell_value(rowx=row_index, colx=3))
    lpeakenergy = int(xl_sheet.cell_value(rowx=row_index, colx=4))
    hpeakenergy = int(xl_sheet.cell_value(rowx=row_index, colx=5))
    label = int(xl_sheet.cell_value(rowx=row_index, colx=6))
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
finalDataRDD = sc.parallelize(finalData)
SolarFlareDF = sqlContext.createDataFrame(finalDataRDD, isSchema)
sqlContext.registerDataFrameAsTable(SolarFlareDF, "solarflare_data")

######################################################################################################

sf_workbook = xlrd.open_workbook('Solar_Flare_Data3.xlsx')
xl_sheet = sf_workbook.sheet_by_index(0)

fldate = []
for row_index in range(1, xl_sheet.nrows):
    row = xl_sheet.row(row_index)
    flareid = int(xl_sheet.cell_value(rowx=row_index, colx=0))
    datemonth = xl_sheet.cell_value(rowx=row_index, colx=1)
    chng2dt = datetime.datetime(*xlrd.xldate_as_tuple(datemonth, sf_workbook.datemode))
    date = int(chng2dt.day)
    month = int(chng2dt.month)
    fldate.append({'flareid': flareid, 'date': date, 'month': month})

isSchema2 = StructType([
                       StructField('flareid', IntegerType(), False),
                        StructField('date', IntegerType(), False),
                        StructField('month', IntegerType(), False)])

fldateRDD = sc.parallelize(fldate)
SolarFlareDateDF = sqlContext.createDataFrame(fldateRDD, isSchema2)
sqlContext.registerDataFrameAsTable(SolarFlareDateDF, "solarflaredate_data")

######################################################################################################

FlareDate_DF = sqlContext.sql("""
            SELECT solarflare_data.dursec as fdur, solarflare_data.peakcpers as fpeakcountps,\
             solarflare_data.totalpeakcnts as ftpcs, solarflare_data.lpeakenergy as flenergy, solarflare_data.hpeakenergy as fhenergy, solarflaredate_data.month as fmonth,\
             solarflaredate_data.date as fdate 
            FROM solarflare_data JOIN solarflaredate_data 
            ON solarflare_data.flareid = solarflaredate_data.flareid
            WHERE solarflare_data.label = 1
            """)
sqlContext.registerDataFrameAsTable(FlareDate_DF, "FlareData")
InstaFlare_DF = sqlContext.sql("""
            SELECT FlareData.ftpcs as ftpcs, FlareData.fdur as fdur, FlareData.fpeakcountps as fpeakcountps, FlareData.flenergy as flenergy, FlareData.fhenergy as fhenergy, instagram_data.month as imonth
            FROM FlareData JOIN instagram_data 
            ON (FlareData.fmonth = instagram_data.month AND  FlareData.fdate = (instagram_data.date + 2))
            """)
sqlContext.registerDataFrameAsTable(InstaFlare_DF, "InstaFlareData")


convert_to_rdd = InstaFlare_DF.rdd

convert_to_rdd = convert_to_rdd.map(lambda l: {'ftpcs': l[0], 'fdur': l[1], 'fpeakcountps': l[2], 'flenergy': l[3], 'fhenergy': l[4], 'imonth': l[5]}).coalesce(1)

rdd_frmt = convert_to_rdd.map(lambda op: json.dumps(op))
rdd_frmt.saveAsTextFile('InstaFlareData11')
