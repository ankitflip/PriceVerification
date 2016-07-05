from library import *
from skylineConstants import *

from AnomalyDetection import AnomalyDetection
from AnomalyDb import AnomalyDb

def removePreAnomaly(anomalyDbObj, inputList, fsn):
	if (len(inputList) == 0):
		return inputList

	retList = []

	for idx, eachTuple in inputList.iterrows():
		
		eachTimeStamp = eachTuple[timeStampName]
		aiDP_Value = eachTuple[aiDP]
		checkDict = {timeStampName:eachTimeStamp, fsnName:fsn, aiDP:aiDP_Value}
		isPresent = anomalyDbObj.checkIfTuplePresent(checkDict)

		if (not isPresent):
			retList.append({timeStampName:eachTimeStamp, aiDP:aiDP_Value})

	retList = pd.DataFrame(retList)

	return retList

def createCheckList(contentDict, anomalyDbObj, fsn, configDict, lastTimeStamp):
	retList = []

	crawlList = contentDict['hits']['hits']
	crawlListLen = len(crawlList)
	notNullAiDP = 0

	latestTimeStamp = int(time.time() * 1000)

	noOfDays = configDict['BasicConfig']['noOfDays']

	for eachCrawl in crawlList:
		eachCrawl = eachCrawl['_source']
		timeStamp = eachCrawl['createdAt']
		listingEags = eachCrawl['listingEags']

		if ((latestTimeStamp-timeStamp > noOfDays*oneDay) or (latestTimeStamp<timeStamp) or (lastTimeStamp<timeStamp)):
			continue

		aiDP_Value = None

		for eachListingEag in listingEags:
			displayedListing = eachListingEag['displayedListing']
			if (displayedListing == True):
				aiDP_Value = eachListingEag['sellerPrice']['value']
				break

		if aiDP_Value:
			tempDict = {timeStampName : timeStamp, aiDP : aiDP_Value}
			retList.append(tempDict)
			notNullAiDP += 1

	retList = pd.DataFrame(retList)

	retList = retList.sort_values(by = [timeStampName], ascending = [True])

	lastEntryRetList = retList.tail(n=1)
	#sys.exit()
	retList = removePreAnomaly(anomalyDbObj, retList.iloc[0:-1], fsn)

	retList = retList.append(lastEntryRetList)
	#print("retList = ")
	#print(retList)

	return (retList, lastEntryRetList[aiDP].iloc[0])

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

def putExceptionInFile(resultDict, resultFilename, createResultFile):
	resultList = []
	resultDictKeys = resultDict.keys()
	colNameList = exceptionColNameList
	for eachCol in colNameList:
		if (eachCol in resultDictKeys):
			resultList.append(resultDict[eachCol])
		else:
			resultList.append(None)

	#print('resultList')
	#print(resultList)

	if (createResultFile):
		resultFile = open(resultFilename, 'w')
		wr = csv.writer(resultFile, quoting=csv.QUOTE_ALL)
		wr.writerow(colNameList)

		resultFile.close()

	resultFile = open(resultFilename, 'a')

	wr = csv.writer(resultFile, quoting=csv.QUOTE_ALL)
	wr.writerow(resultList)

	resultFile.close()


def notifKafkaConsumer():

	inputDF = pd.read_csv('../Config1.5-5/aiDPHistoryResult.csv', index_col=False, header=0)
	
	noOfAiNotif = noOfExceptions = 0
	configFilename = '../Config1.5-5/checkForAnomaly.yml'
	with open(configFilename, 'r') as configFile:
		configDict = yaml.load(configFile)

	anomalyDbObj = AnomalyDb(configDict)
	anomalyDbObj.dbConnect()

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

			#time.sleep(1.0/1000)
			#productId = 'EMLDGKUHYQM3PDPF'
			contents = urllib.urlopen("http://10.47.4.1/kb/v1.0/products/mappings/domainId/flipkart.com/productId/"+fsn+"/targetDomain/amazon.in").read()
			contents = json.loads(contents)
			endTime = time.clock()

			startTime = time.clock()
			comId = contents['productId']
			url = 'http://10.33.237.22:9200/product_listings_eag/product_listings_eag_type/_search?size=150'
			payload = '{"query": {"bool": {"must": [{"match": {"productId" : "'+comId+'"}}]}},"sort": [{"createdAt": {"order": "desc"}}]}'
			content = urllib.urlopen(url, payload).read()
			content = json.loads(content)
			endTime = time.clock()
			#print("aiContent  time elapsed = ", endTime-startTime)

			print(noOfAiNotif)
			(checkList, aiDP_Value) = createCheckList(content, anomalyDbObj, fsn, configDict, timeStamp)

			verValue = eachRowInputDF[verAttrName]
			com = eachRowInputDF[comName]

			#print(checkList)
			#sys.exit()
			anomalyDetector = AnomalyDetection()

			(retFlag, retDict) = anomalyDetector.new_checkForAnomaly(checkList, configDict)
			
			isAnomaly = retFlag
			noOfData = len(checkList)
			
			resultFileDict = {comName : com, timeStampName : timeStamp, verAttrName : verValue, fsnName : fsn, aiDP : aiDP_Value, anomalyAttrName : isAnomaly, noOfDataAttrName : noOfData}
			resultFileDict.update(retDict)

			print(resultFileDict)

			resultFilename = '../Config1.5-5/aiDPHistoryResultHistChanged.csv'
			createResultFile = (noOfAiNotif == 1)
			#print('noOfAiNotif = ', noOfAiNotif)
			putResultInFile(resultFileDict, resultFilename, createResultFile)
		
		except Exception as e:
			noOfExceptions += 1
			print('Exception occured')
			print('fsn = ', fsn)
			print(e)
		


if __name__ == '__main__':
	notifKafkaConsumer()