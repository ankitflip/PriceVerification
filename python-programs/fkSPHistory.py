from library import *
from skylineConstants import *

from AnomalyDetection import AnomalyDetection

def createFkSPCheckList(crawlList, fsn, configDict):
	retList = []

	crawlListLen = len(crawlList)
	notNullAiDP = 0

	latestTimeStamp = int(time.time() * 1000)

	noOfDays = configDict['BasicConfig']['noOfDays']

	for eachCrawl in crawlList:
		timeStamp = eachCrawl[timeStampName]
		'''
		pattern = '%Y-%m-%d %H:%M:%S' #'2016-05-05 23:55:42.0'
		timeStamp = int(time.mktime(time.strptime(timeStamp, pattern)))
		print(timeStamp)
		'''

		fkSP_Value = None

		if 'fsp' in eachCrawl.keys():
			fkSP_Value = eachCrawl['fsp']

		if fkSP_Value:
			tempDict = {timeStampName : timeStamp, aiDP : fkSP_Value}
			retList.append(tempDict)
			notNullAiDP += 1

	retList = pd.DataFrame(retList)

	retList = retList.sort_values(by = [timeStampName], ascending = [True])

	return retList


def putResultInFile(resultDict, resultFilename, createResultFile):
	resultList = []
	resultDictKeys = resultDict.keys()
	for eachCol in resultColNameList:
		if (eachCol in resultDictKeys):
			resultList.append(resultDict[eachCol])
		else:
			resultList.append(None)

	#print('resultList')
	#print(resultList)

	if (createResultFile):
		resultFile = open(resultFilename, 'w')
		wr = csv.writer(resultFile, quoting=csv.QUOTE_ALL)
		wr.writerow(resultColNameList)

		resultFile.close()

	resultFile = open(resultFilename, 'a')

	wr = csv.writer(resultFile, quoting=csv.QUOTE_ALL)
	wr.writerow(resultList)

	resultFile.close()


def fkSPHistory():

	inputDF = pd.read_csv('../Config1.5-5/inputForFkSP.csv', index_col=False, header=0)
	
	noOfAiNotif = noOfExceptions = 0
	configFilename = '../Config1.5-5/checkForAnomaly.yml'
	with open(configFilename, 'r') as configFile:
		configDict = yaml.load(configFile)

	preFsn = preTimeStamp = 0

	for idx, eachRowInputDF in inputDF.iterrows():
		try:
			noOfAiNotif += 1
			fsn = eachRowInputDF[fsnName]
			timeStamp = eachRowInputDF[timeStampName]

			if ((fsn == preFsn) and (timeStamp == preTimeStamp)):
				continue

			preFsn = fsn
			preTimeStamp = timeStamp

			time.sleep(1.0/30)

			url = 'http://10.47.4.7/priceHistory/getLimited'
			payload = '{"queryParams":{"fsn":"'+fsn+'","from":"2016-03-10T22:24:04","to":"2016-06-27T14:41:12"}, "limit" : 50}'
			req = urllib.Request(url)
			req.add_header('Content-Type', 'application/json; charset=utf-8')
			req.add_header('Content-Length', len(payload))
			req.add_header('X-Flipkart-Client', 'ci_client')
			latestTimeStamp = int(time.time() * 1000)
			req.add_header('X-Request-ID', latestTimeStamp)
			content = urllib.urlopen(req, payload).read()
			content = json.loads(content)
			#print(content)
			print(noOfAiNotif)
			print(latestTimeStamp)
			checkList = createFkSPCheckList(content, fsn, configDict)

			verValue = eachRowInputDF[verAttrName]
			com = eachRowInputDF[comName]
			aiDP_Value = eachRowInputDF[aiDP]

			tempList = [{timeStampName : timeStamp, aiDP : aiDP_Value}]
			#print(tempList)
			checkList = checkList.append(pd.DataFrame(tempList))
			#print(checkList)
			#sys.exit()
			anomalyDetector = AnomalyDetection()

			(retFlag, retDict) = anomalyDetector.new_checkForAnomaly(checkList, configDict)
			
			isAnomaly = retFlag
			noOfData = len(checkList)
			
			resultFileDict = {comName : com, timeStampName : timeStamp, verAttrName : verValue, fsnName : fsn, aiDP : aiDP_Value, anomalyAttrName : isAnomaly, noOfDataAttrName : noOfData}
			resultFileDict.update(retDict)

			print(resultFileDict)

			resultFilename = '../Config1.5-5/fkSPHistoryResult.csv'
			createResultFile = (noOfAiNotif == 1)
			#print('noOfAiNotif = ', noOfAiNotif)
			putResultInFile(resultFileDict, resultFilename, createResultFile)
		
		except Exception as e:
			noOfExceptions += 1
			print('Exception occured')
			print('fsn = ', fsn)
			print(e)
		

if __name__ == '__main__':
	fkSPHistory()