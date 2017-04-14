Step 1: Run the sfclassify_skl.py with the Solar_Flare_Data3.xlsx file. This file saves data with labels used in next step[Data.xlsx].
Step 2: Run the JavaScript file and instascrape.py file to get Instagram links [this takes a lot of time, so we have included our links and data that we scraped using both those scripts. They are names: hashtagdata.txt's and InstagramFinalData.txt]
Step 3: Run sf_insta_integration.py and that provides with the output file: InstaFlareData.txt 
Step 4: Run flareinsta_baggingestimator.py to get the final clusters and their counts.

Experiments:

Run the sfclassify_mllib.py with the Solar_Flare_Data3.xlsx file. 
Run the flareinsta_svm.py with InstaFlareData.txt 