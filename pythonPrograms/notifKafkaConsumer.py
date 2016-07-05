from library import *
from skylineConstants import *

from AnomalyDetection import AnomalyDetection
from AnomalyDb import AnomalyDb


class NotifKafkaConsumer(object):

	def removePreAnomaly(self, anomalyDbObj, inputList, fsn):
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

	def createCheckList(self, contentDict, anomalyDbObj, fsn, configDict):
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

			if ((latestTimeStamp - timeStamp > noOfDays * oneDay) or (latestTimeStamp < timeStamp)):
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
		
		if (retList.empty):
			raise ValueError('Zero Data Points in History KB')

		retList = retList.sort_values(by = [timeStampName], ascending = [True])

		lastEntryRetList = retList.tail(n=1)
		retList = self.removePreAnomaly(anomalyDbObj, retList.iloc[0:-1], fsn)

		retList = retList.append(lastEntryRetList)

		return retList

	def putResultInFile(self, resultDict, resultFilename, createResultFile):
		resultList = []
		resultDictKeys = resultDict.keys()
		for eachCol in resultColNameList:
			if (eachCol in resultDictKeys):
				resultList.append(resultDict[eachCol])
			else:
				resultList.append(None)

		if (createResultFile):
			resultFile = open(resultFilename, 'w')
			wr = csv.writer(resultFile, quoting=csv.QUOTE_ALL)
			wr.writerow(resultColNameList)

			resultFile.close()

		resultFile = open(resultFilename, 'a')

		wr = csv.writer(resultFile, quoting=csv.QUOTE_ALL)
		wr.writerow(resultList)

		resultFile.close()

	def putExceptionInFile(self, resultDict, resultFilename, createResultFile):
		resultList = []
		resultDictKeys = resultDict.keys()
		colNameList = exceptionColNameList
		for eachCol in colNameList:
			if (eachCol in resultDictKeys):
				resultList.append(resultDict[eachCol])
			else:
				resultList.append(None)

		if (createResultFile):
			resultFile = open(resultFilename, 'w')
			wr = csv.writer(resultFile, quoting=csv.QUOTE_ALL)
			wr.writerow(colNameList)

			resultFile.close()

		resultFile = open(resultFilename, 'a')

		wr = csv.writer(resultFile, quoting=csv.QUOTE_ALL)
		wr.writerow(resultList)

		resultFile.close()


	def solve(self):
		consumer = KafkaConsumer('ci_signal_pars', bootstrap_servers=['10.33.229.22:9092', '10.33.165.2:9092', '10.33.197.163:9092'])#, api_version= '0.8.2')
		#consumer = ['ACCEE9M6BSAJEF2P']
		noOfExceptions = 0
		noOfAiNotif = 0
		configFolder = '../Config1.5-5/'
		configFilename = configFolder+'checkForAnomaly.yml'
		with open(configFilename, 'r') as configFile:
			configDict = yaml.load(configFile)

		anomalyDbObj = AnomalyDb(configDict)
		anomalyDbObj.dbConnect()

		for notif in consumer:
			try:
				notif = json.loads(notif.value)
				#notif = {u'parsData': {u'geoInformation': {u'city': None, u'state': None, u'pincode': u'700091', u'zone': None, u'country': None}, u'displayedListing': {u'shippings': [], u'price': {u'dealPrice': None, u'mrp': None, u'sellingPrice': {u'currency': u'INR', u'value': 189.0}}, u'availability': {u'availabilityType': u'IN_STOCK', u'paymentModes': [], u'sla': None, u'isFulfilledByCompetitor': False}}, u'otherListings': []}, u'validTill': 1466937187374, u'competitor': u'AI', u'vertical': u'cases_covers', u'fetchTimestamp': 1466749987374, u'channel': u'WEBSITE', u'productId': u'TABDPZRVEKXBBKC7'}
				com = notif[comName]

				if (com != comAi):
					continue 
				noOfAiNotif += 1

				timeStamp = notif['fetchTimestamp']
				verValue = notif[verAttrName]
				fsn = notif['productId']
				
				#productId = 'EMLDGKUHYQM3PDPF'
				contents = urllib.urlopen("http://10.47.4.1/kb/v1.0/products/mappings/domainId/flipkart.com/productId/"+fsn+"/targetDomain/amazon.in").read()
				contents = json.loads(contents)

				comId = contents['productId']
				url = 'http://10.33.237.22:9200/product_listings_eag/product_listings_eag_type/_search?size=150'
				payload = '{"query": {"bool": {"must": [{"match": {"productId" : "'+comId+'"}}]}},"sort": [{"createdAt": {"order": "desc"}}]}'
				content = urllib.urlopen(url, payload).read()
				content = json.loads(content)

				checkList = self.createCheckList(content, anomalyDbObj, fsn, configDict)
				anomalyDetector = AnomalyDetection()

				(retFlag, retDict) = anomalyDetector.new_checkForAnomaly(checkList, configDict)
				
				aiDP_Value = checkList.tail(1).get_value(0, aiDP)
				isAnomaly = retFlag
				noOfData = len(checkList)

				resultFileDict = {comName : com, timeStampName : timeStamp, verAttrName : verValue, fsnName : fsn, aiDP : aiDP_Value, anomalyAttrName : isAnomaly, noOfDataAttrName : noOfData}
				resultFileDict.update(retDict)
				resultFilename = configFolder+'resultFile.csv'
				createResultFile = (noOfAiNotif == 1)
				self.putResultInFile(resultFileDict, resultFilename, False) #createResultFile)

				if (isAnomaly):
					anomalyDbObj.insert(resultFileDict)

				print(resultFileDict)
			
			except Exception as e:
				noOfExceptions += 1
				print('Exception occured')
				print('Kafka notif = ')
				print(notif)
				print(e)
				#traceback.print_exc()
				exceptionFileDict = {notifName:notif, exceptionName:e}
				exceptionFilename = configFolder+'exception.csv'
				createExceptionFile = (noOfExceptions == 1)
				self.putExceptionInFile(exceptionFileDict, exceptionFilename, False) #createExceptionFile)

		anomalyDbObj.dbClose()
		print('noOfExceptions = %d'%(noOfExceptions))
			

if __name__ == '__main__':
	NotifKafkaConsumer().solve()