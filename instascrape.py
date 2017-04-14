import requests
import xlrd
import json
from bs4 import BeautifulSoup
import sys

from pyspark import SparkConf, SparkContext, SQLContext

outputpath = sys.argv[1]
sf_workbook = xlrd.open_workbook('Links1_5.xlsx')
xl_sheet = sf_workbook.sheet_by_index(0)

conf = SparkConf().setAppName('PBD2 Project')
sc = SparkContext(conf=conf)

alldata = []
for row_index in range(1, xl_sheet.nrows):
    print row_index
    link = xl_sheet.row(row_index)

    link = str(xl_sheet.cell_value(rowx=row_index, colx=0))
    
    req = requests.get(link).text
    soup = BeautifulSoup(req, "html.parser")

    titlename = soup.title.string
    boolvalue = "Not" in titlename

    if(boolvalue == False):
        
            
        for meta in soup.findAll("meta"):
                metaname = meta.get('name', '').lower()
                metaprop = meta.get('property', '').lower()
                if metaprop.find("title")>0:
                    datedata = meta['content'].strip()
                    
        scripts = soup.find_all("script")
        for script in scripts:
                if script.text[:18] == "window._sharedData":
                        break

        data = json.loads(script.contents[0][21:-1])
        urllink = str(data["entry_data"]["PostPage"][0]["media"]["display_src"])
        locationdata = str(data["entry_data"]["PostPage"][0]["media"]["location"])
        
        if(data["entry_data"]["PostPage"][0]["media"]["is_video"]):
                
                videoflag = 1
        else:
                videoflag = 0
                
        alldata.append({'metadata': datedata, 'location': locationdata,\
         'isvideo': videoflag, 'urllink': urllink})          
        
print alldata
finalDataRDD = sc.parallelize(alldata).coalesce(1)
outdata = finalDataRDD.map(lambda op: json.dumps(op))

outdata.saveAsTextFile(outputpath)

