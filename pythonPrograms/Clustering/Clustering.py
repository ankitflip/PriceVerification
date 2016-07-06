import sys, re, json
import numpy as np
import matplotlib.pyplot as plt
from sets import Set
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import subprocess
import math

eps = 1e-5#math.pow(10, -5)
epsNull = 1e-4
noOfAttr = 5

''' 
zero = 0

fsnName = fsn
timeStampName = timestamp
aiLSCWF - ai_lowest_shipping_charge_without_fba
aiMrpName - ai_mrp
mrpName - mrp
aiADSP = ai_displayed_selling_price
aiLSP = ai_lowest_selling_price
aiLSPWF = ai_lowest_selling_price_without_fba
aiLP = ai_lowest_price
aiDP = ai_displayed_price
fkSP = flipkart_selling_price
aiADSP_diff_fkSP_Name = aiADSP_diff_fkSP_Name
aiADSP_diff_mrpName_Name = aiADSP_diff_mrpName_Name

shippingDaysName = shipping_days
aiMinSlaName = ai_min_sla
aiMaxSlaName = ai_max_sla
aiFbaAttrName = ai_fba
'''

'''
#new features starts
aiADSP_diff_fkSP_Value = (aiADSP_Value - fkSP_Value)/fkSP_Value
aiADSP_diff_mrpName_Value = (aiADSP_Value - mrpName_Value)/mrpName_Value
#new features ends
'''

zero = 0

fsnName = 'fsn'
timeStampName = 'timestamp'
aiFbaAttrName = 'ai_fba'
aiLSCWFName = 'ai_lowest_shipping_charge_without_fba'
aiMrpName = 'ai_mrp'
mrpName = 'mrp'
aiADSP = 'ai_displayed_selling_price'
aiLSP = 'ai_lowest_selling_price'
aiLSPWF = 'ai_lowest_selling_price_without_fba'
aiLP = 'ai_lowest_price'
aiDP = 'ai_displayed_price'
fkSP = 'flipkart_selling_price'

shippingDaysName = 'shipping_days'
aiMinSlaName = 'ai_min_sla'
aiMaxSlaName = 'ai_max_sla'


allAttrList = [timeStampName, mrpName, aiMrpName, aiADSP, aiLSP, aiLSPWF, aiLP, aiDP, fkSP, aiFbaAttrName, shippingDaysName, aiMinSlaName, aiMaxSlaName]
allNumericAttr = [mrpName, aiMrpName, aiADSP, aiLSP, aiLSPWF, aiLP, aiDP, fkSP, shippingDaysName, aiMinSlaName, aiMaxSlaName]
reqOneOfAttr = [[timeStampName], [aiADSP, aiLSP, aiLSPWF, aiLP, aiDP, fkSP], [mrpName, aiMrpName], [shippingDaysName, aiMinSlaName, aiMaxSlaName]]
aiPriceAttrList = [aiADSP, aiLSP, aiLSPWF, aiLP, aiDP, fkSP]
clusterAttrList = [0, 5] #[0, 1, 2, 3, 4, 10] 

trainAttrList = ['aiADSP_diff_fkSP_Value', 
				 'aiAvgSla_Value',
				 'aiFbaAttr_Value',
				 'aiADSP_diff',
				 'aiADSP_diff_percent',
				 'aiADSP_Value',
				 'fkSP_Value',
				 'mrpName_Value',
				 'fsn_Value',
				 'timeStamp_Value'
				]

colors = ['g.', 'r.', 'c.', 'b.', 'm.', 'y.', 'k.', 'g.', 'r.']

def makeClusterKMeans(valueListFile):
	valueListFile = open(valueListFile, 'r')
	valueList = json.load(valueListFile)
	#valueList = valueList
	#trainValueList = valueList[:, 0:noOfAttr]
	trainValueList = []
	noOfValueRow = len(valueList)
	for row in range(noOfValueRow):
		tempList = []
		for col in range(noOfAttr):
			tempList.append(valueList[row][col])
		trainValueList.append(tempList)

	trainValueList = np.array(trainValueList)

	print('len of trainValueList = ', len(trainValueList[0]))

	noOfTrainRows = len(trainValueList)

	#attrValueList :- train attribute value list
	attrValueList = []

	for col in range(noOfAttr):
		attrValueList.append([])

	for row in range(noOfTrainRows):
		for col in range(noOfAttr):
			attrValueList[col].append(trainValueList[row][col])

	attrValueMax = []
	for col in range(noOfAttr):
		attrValueMax.append(max(attrValueList[col]))
		
	for row in range(noOfTrainRows):
		for col in range(noOfAttr):
			trainValueList[row][col] /= attrValueMax[col]

	for row in range(noOfTrainRows):
		if (abs(1 - trainValueList[row][0]) < .1):
			print(valueList[row])

	#print('noOfAttr = ', noOfAttr)

	#sys.exit()

	#plt.hist(trainValueList[:,0], bins=100)
	#plt.show()

	noOfClusters = 2
	kmeans = KMeans(n_clusters = noOfClusters)
	kmeans.fit(trainValueList)

	centroids = kmeans.cluster_centers_
	labels = kmeans.labels_

	
	print('centroids = ', centroids)
	print('labels = ', labels)

	colors = ['g.', 'r.', 'c.', 'b.']

	for i in range(len(trainValueList)):
		plt.plot(trainValueList[i][0], trainValueList[i][2], colors[labels[i]], markersize = 10)

	plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'x', s = 150, linewidths = 5, zorder = 10)

	plt.show()
	

def inputDumpFileProcessing(inputFilename, outputFilename):
	inputFile = open(inputFilename, 'r')
	#outputFile = open(outputFilename, 'w')

	#-----Creating fsnList----------
	tempInputFile = open(inputFilename, 'r')
	fsnList = []
	for line in tempInputFile:
		fsnLineObj =  re.search(r'column=data:fsn.+', line)

		if fsnLineObj:
			fsnLine = fsnLineObj.group()
			fsnList.append(re.search(r'value=.+', fsnLine).group()[len('value='):])
	tempInputFile.close()

	outputArray = {}
	for eachFsn in fsnList:
		outputArray[eachFsn] = {}

	verFsnDict = {}

	#WSAI : whiteSpaceArrayIndex
	WSAI = {'fsn':0, 'attr':1, 'timeStamp':2, 'attrValue':3}
	i = 0
	for line in inputFile:
		i += 1
		try:
			whiteSpaceArray = re.findall(r'\S+', line)
			'''
			For whiteSpaceArray index:
				0 -> 2013-07-16T14:50:02.711Z-130-ACCCRHTNBKYE4JYZ
				1 -> column=data:ai_crawl_status,
				2 -> timestamp=1373986578668,
				3 -> value=Partial
			'''
			if ((not whiteSpaceArray) or (len(whiteSpaceArray) != 4)):
				continue
			columnCheck = re.search(r'column=.+', whiteSpaceArray[WSAI['attr']])
			if not columnCheck:
				continue
			timeStampCheck = re.search(r'timestamp=.+', whiteSpaceArray[WSAI['timeStamp']])
			if not timeStampCheck:
				continue
			valueCheck = re.search(r'value=.+', whiteSpaceArray[WSAI['attrValue']])
			if not valueCheck:
				continue

			fsn = re.findall(r'-\w+', whiteSpaceArray[WSAI['fsn']])[-1][1:]
			attr = re.search(r':\w+', whiteSpaceArray[WSAI['attr']]).group()[1:]
			timeStamp = re.search(r'=\w+', whiteSpaceArray[WSAI['timeStamp']]).group()[1:]
			attrValue = re.search(r'=.+', whiteSpaceArray[WSAI['attrValue']]).group()[1:]
			if fsn not in outputArray:
				continue
			
			if timeStamp not in outputArray[fsn]:
				outputArray[fsn][timeStamp] = {attr:attrValue}
			else:
				outputArray[fsn][timeStamp][attr] = attrValue
			
			if attr == 'vertical':
				if attrValue not in verFsnDict:
					verFsnDict[attrValue] = set()
				verFsnDict[attrValue].add(fsn)
		except Exception as e:
			print(i)
			print(line)
			print(sys.exc_info() )
			sys.exit()
	
	#json.dump(outputArray, outputFile)

	verOutputDict = {}
	allVerList = verFsnDict.keys()
	for eachVer in allVerList:
		verOutputDict[eachVer] = {}
		for tempFsn in verFsnDict[eachVer]:
			verOutputDict[eachVer][tempFsn] = outputArray[tempFsn]
		outputFile = open('../verticals/'+eachVer, 'w')
		json.dump(verOutputDict[eachVer], outputFile)

	#print(verOutputDict)

	inputFile.close()
	outputFile.close()

def processOnEachVertical():
	inputFile = open('../verticals/mobile', 'r')
	inputDict = json.load(inputFile)

	fsnList = inputDict.keys()
	noOfAi = notAvailable = isAvailable = noOfNull = noOfRowsToSkip = noOfZeros = noOfTrain = noOfaiAvgSla_LessThanEps = 0

	aiSet = Set()

	aiFbaAttrName = 'ai_fba'
	aiFbaNumeric = {'false':-1, 'true':1}

	''' aiLSCWF - ai_lowest_shipping_charge_without_fba
		aiMrpName - ai_mrp
		mrpName - mrp
		aiADSP = ai_displayed_selling_price
		aiLSP = ai_lowest_selling_price
		aiLSPWF = ai_lowest_selling_price_without_fba
		aiLP = ai_lowest_price
		aiDP = ai_displayed_price
		fkSP = flipkart_selling_price
	'''
	aiLSCWFName = 'ai_lowest_shipping_charge_without_fba'
	aiMrpName = 'ai_mrp'
	mrpName = 'mrp'
	aiADSP = 'ai_displayed_selling_price'
	aiLSP = 'ai_lowest_selling_price'
	aiLSPWF = 'ai_lowest_selling_price_without_fba'
	aiLP = 'ai_lowest_price'
	aiDP = 'ai_displayed_price'
	fkSP = 'flipkart_selling_price'
	aiADSP_diff_fkSP_Name = 'aiADSP_diff_fkSP_Name'
	aiADSP_diff_mrpName_Name = 'aiADSP_diff_mrpName_Name'

	shippingDaysName = 'shipping_days'
	aiMinSlaName = 'ai_min_sla'
	aiMaxSlaName = 'ai_max_sla'

	'''
	#new features starts
	aiADSP_diff_fkSP_Value = (aiADSP_Value - fkSP_Value)/fkSP_Value
	aiADSP_diff_mrpName_Value = (aiADSP_Value - mrpName_Value)/mrpName_Value
	#new features ends
	'''

	trainAttr = [aiADSP_diff_fkSP_Name, aiADSP_diff_mrpName_Name]
	trainValueList = []
	train_aiADSP_diff_fkSP_List = []
	train_aiADSP_diff_mrpName_List = []

	allAttrList = [mrpName, aiMrpName, aiADSP, aiLSP, aiLSPWF, aiLP, aiDP, fkSP, aiFbaAttrName, shippingDaysName, aiMinSlaName, aiMaxSlaName]
	allNumericAttr = [mrpName, aiMrpName, aiADSP, aiLSP, aiLSPWF, aiLP, aiDP, fkSP, shippingDaysName, aiMinSlaName, aiMaxSlaName]
	reqOneOfAttr = [[aiADSP, aiLSP, aiLSPWF, aiLP, aiDP, fkSP], [mrpName, aiMrpName], [shippingDaysName, aiMinSlaName, aiMaxSlaName]]
	aiPriceAttrList = [aiADSP, aiLSP, aiLSPWF, aiLP, aiDP, fkSP]

	for eachFsn in fsnList:
		#outputFile = open('blah', 'w')
		timeStampList = inputDict[eachFsn].keys()
		#try:
		for eachTimeStampList in timeStampList:
			#json.dump(inputDict[eachFsn][eachTimeStampList], outputFile)
			competitorName = inputDict[eachFsn][eachTimeStampList]['competitor']
			
			#Checking for Amazon Products
			if (competitorName == 'ai'):
				noOfAi += 1
				allAttr = inputDict[eachFsn][eachTimeStampList].keys()
				
				
				#Checking for aiFbaAttrName starts
				if (aiFbaAttrName in allAttr):
					inputDict[eachFsn][eachTimeStampList][aiFbaAttrName] = aiFbaNumeric[inputDict[eachFsn][eachTimeStampList][aiFbaAttrName]]
				else:
					inputDict[eachFsn][eachTimeStampList][aiFbaAttrName] = 0.5
				#Checking for aiFbaAttrName ends
				
				
				#All the values must be null or numeric or numeric string
				#Changing all values to numeric values .. and after this step all attr will be present 
				for attrName in allAttrList:
					if ( (not(attrName in allAttr)) or (inputDict[eachFsn][eachTimeStampList][attrName] == 'null')):
						inputDict[eachFsn][eachTimeStampList][attrName] = 0
					else:
						inputDict[eachFsn][eachTimeStampList][attrName] = float(inputDict[eachFsn][eachTimeStampList][attrName])

				allAttr = inputDict[eachFsn][eachTimeStampList].keys()

				reqOneOfAttrFlag = 0

				for i in range(len(reqOneOfAttr)):
					reqOneOfAttrFlag = 0
					for eachReqOneOfAttr in reqOneOfAttr[i]:
						if (abs(inputDict[eachFsn][eachTimeStampList][eachReqOneOfAttr] - 0) > eps):
							reqOneOfAttrFlag = 1
							break
					if (reqOneOfAttrFlag == 0):
						break

				if (reqOneOfAttrFlag == 0):
					noOfRowsToSkip += 1
					continue


				noOfTrain += 1

				for i in range(len(reqOneOfAttr)):
					allAvg = allSum = allCnt = 0.0
					for eachReqOneOfAttr in reqOneOfAttr[i]:
						if (abs(inputDict[eachFsn][eachTimeStampList][eachReqOneOfAttr] - 0) > eps):
							allSum += inputDict[eachFsn][eachTimeStampList][eachReqOneOfAttr]
							allCnt += 1
					allAvg = allSum/allCnt
					for eachReqOneOfAttr in reqOneOfAttr[i]:
						if (not (abs(inputDict[eachFsn][eachTimeStampList][eachReqOneOfAttr] - 0) > eps)):
							inputDict[eachFsn][eachTimeStampList][eachReqOneOfAttr] = allAvg

				
				aiADSP_Value = inputDict[eachFsn][eachTimeStampList][aiADSP]
				fkSP_Value = inputDict[eachFsn][eachTimeStampList][fkSP]
				mrpName_Value = inputDict[eachFsn][eachTimeStampList][mrpName]
				aiFbaAttr_Value = inputDict[eachFsn][eachTimeStampList][aiFbaAttrName]

				aiMinSla_Value = inputDict[eachFsn][eachTimeStampList][aiMinSlaName]
				aiMaxSla_Value = inputDict[eachFsn][eachTimeStampList][aiMaxSlaName]

				try:
					tempList = []			

					aiADSP_diff_fkSP_Value = (aiADSP_Value - fkSP_Value)/mrpName_Value#fkSP_Value
					aiADSP_diff_mrpName_Value = (aiADSP_Value - mrpName_Value)/mrpName_Value
					aiAvgSla_Value = (aiMinSla_Value + aiMaxSla_Value)/2
					if(abs(aiAvgSla_Value - 0) <= eps):
						noOfaiAvgSla_LessThanEps += 1

					tempList.append(aiADSP_diff_fkSP_Value)
					tempList.append(aiAvgSla_Value)#aiFbaAttr_Value)#aiADSP_diff_mrpName_Value)
					tempList.append(aiFbaAttr_Value)

					tempList.append(aiADSP_Value)
					tempList.append(fkSP_Value)
					tempList.append(mrpName_Value)
					tempList.append(eachFsn)

					#train_aiADSP_diff_fkSP_List.append(aiADSP_diff_fkSP_Value)
					#train_aiADSP_diff_mrpName_List.append(aiADSP_diff_mrpName_Value)

					trainValueList.append(tempList)
				except:
					print(inputDict[eachFsn][eachTimeStampList])
					
					print('----------------------------------------')
					for i in range(len(reqOneOfAttr)):
						for eachReqOneOfAttr in reqOneOfAttr[i]:
							print(eachReqOneOfAttr)
							print(inputDict[eachFsn][eachTimeStampList][eachReqOneOfAttr])
							print( (abs(inputDict[eachFsn][eachTimeStampList][eachReqOneOfAttr] - 0) > eps))
					print(fkSP_Value)
					print(mrpName_Value)
					for eachAttr in allAttrList:
						print(eachAttr)
						print(inputDict[eachFsn][eachTimeStampList][eachAttr])
					
					sys.exit()

		#outputFile.close()

	print(noOfAi)
	print(noOfTrain)
	print(noOfaiAvgSla_LessThanEps)
	#print(noOfNull)
	#print(noOfRowsToSkip)
	#print(noOfZeros)
	#print(isAvailable)
	#print(notAvailable)
	#print(aiSet)
	trainValueListFile = open('../trainValueListFile', 'w')
	json.dump(trainValueList, trainValueListFile)
	
	trainValueListFile.close()
	inputFile.close()


# new starts


def new_inputDumpFile_to_cleaned(inputFilename, outputFilename, outputFileAttachType):
	inputFile = open(inputFilename, 'r')
	outputFile = open(outputFilename, outputFileAttachType)

	#WSAI : whiteSpaceArrayIndex
	WSAI = {'fsn':0, 'attr':1, 'timeStamp':2, 'attrValue':3}
	i = 0
	for line in inputFile:
		i += 1
		try:
			whiteSpaceArray = re.findall(r'\S+', line)
			'''
			For whiteSpaceArray index:
				0 -> 2013-07-16T14:50:02.711Z-130-ACCCRHTNBKYE4JYZ
				1 -> column=data:ai_crawl_status,
				2 -> timestamp=1373986578668,
				3 -> value=Partial
			'''
			#checking if line of format of cleanedInputDumpFile
			if ((not whiteSpaceArray) or (len(whiteSpaceArray) < 4)):
				#print(line)
				continue
			
			columnCheck = re.search(r'column=.*', whiteSpaceArray[WSAI['attr']])
			if not columnCheck:
				#print('columnCheck,  ', line)
				continue
			timeStampCheck = re.search(r'timestamp=.*', whiteSpaceArray[WSAI['timeStamp']])
			if not timeStampCheck:
				#print('timeStampCheck   ', line)
				continue
			valueCheck = re.search(r'value=.*', whiteSpaceArray[WSAI['attrValue']])
			if not valueCheck:
				#print('valueCheck    ', line)
				continue

			outputFile.write(line)
			
		except Exception as e:
			print(i)
			print(line)
			print(sys.exc_info() )
			sys.exit()

	inputFile.close()
	outputFile.close()

def new_cleaned_to_fsnInputFile(inputFilename, outputFilename, outputFileAttachType, lineFormat):
	inputFile = open(inputFilename, 'r')
	outputFile = open(outputFilename, outputFileAttachType)

	fsnName = 'fsn'
	attrName = 'attr'
	timeStampName = 'timeStamp'
	attrValueName = 'attrValue'

	fsn = 'start Fsn'
	preFsn = 'start PreFsn'
	timeStamp = 'start timeStamp'
	preTimeStamp = 'start preTimeStamp'

	primaryKey = ''
	prePrimaryKey = ''

	noOfClash = noOfCompetitors = 0

	#WSAI : whiteSpaceArrayIndex
	WSAI = {'fsn':0, 'attr':1, 'timeStamp':2, 'attrValue':3}
	searchPattern = {attrName:'column=data:', timeStampName:'timestamp=', attrValueName:'value='}
	lineNo = different = 0
	for line in inputFile:
		lineNo += 1

		whiteSpaceArray = re.findall(r'\S+', line)
		try:
			primaryKey = whiteSpaceArray[WSAI['fsn']]
			if (lineFormat == 1):
				fsn = re.findall(r'-\w+', whiteSpaceArray[WSAI['fsn']])[-1][1:]
			elif (lineFormat == 2):
				fsn = re.search(r'\w+', whiteSpaceArray[WSAI['fsn']]).group()
			#attr = re.search(r':\w+', whiteSpaceArray[WSAI['attr']]).group()[1:]
			attr = re.search(r'column=data:.*', whiteSpaceArray[WSAI['attr']]).group()[len(searchPattern[attrName]):]
			attr = attr[0:len(attr)-1]
			#timeStamp = re.search(r'=\w+', whiteSpaceArray[WSAI['timeStamp']]).group()[1:]
			timeStamp = re.search(r'timestamp=.*', whiteSpaceArray[WSAI['timeStamp']]).group()[len(searchPattern[timeStampName]):]
			timeStamp = timeStamp[0:len(timeStamp)-1]
			attrValue = re.search(r'value=.*', line)#whiteSpaceArray[WSAI['attrValue']]).group()[1:]
			if ((not attrValue) or (attrValue == '') or (attrValue == '\n')):
				attrValue = 'null'
			else:
				attrValue = attrValue.group()[len(searchPattern[attrValueName]):]
				if (len(attrValue)==0):
					attrValue = 'null'

			if (lineNo == 1):
				tempDict = {}
			elif (primaryKey != prePrimaryKey):#((fsn != preFsn) or (timeStamp != preTimeStamp)):
				different += 1
				tempList = {'1':[preFsn, preTimeStamp], '0':tempDict}
				json.dump(tempList, outputFile)
				outputFile.write('\n')
				tempDict = {}

			if (attr == 'competitor'):
				noOfCompetitors += 1
				if(attr in tempDict.keys()):
					noOfClash += 1
			tempDict[attr] = attrValue
			preFsn = fsn
			preTimeStamp = timeStamp
			prePrimaryKey = primaryKey

			'''
			if (lineNo < 50):
				print(line)
				print('timeStamp = %s'%timeStamp)
				print('attr = %s'%attr)
				print('attrValue = %s'%attrValue)
			'''
			
		except Exception as e:
			print (line)
			print ('fsn = ', fsn, '   timeStamp = ', timeStamp, '   preFsn = ', preFsn, '    preTimeStamp = ', preTimeStamp)
			print(sys.exc_info() )
			print ('different = ', different)
			print ('line No = ', lineNo)
			#sys.exit()

	tempList = {'1':[preFsn, preTimeStamp], '0':tempDict}
	json.dump(tempList, outputFile)
	outputFile.write('\n')

	outputFile.close()
	inputFile.close()

	#print("primaryKey = %s"%primaryKey)
	#print("noOfCompetitors = %d"%noOfCompetitors)
	#print("noOfClash = %d"%noOfClash)
	print("last lineNo = %d"%lineNo)
	
'''
verticalClassifier() will take a file which will contain data of all the verticals and change into individual
vertical files
'''
def new_verticalClassifier(inputFilename, outputDir):
	inputFile = open(inputFilename, 'r')
	
	verticalName = 'vertical'
	noOfNoVertical = 0

	for line in inputFile:
		tempList = json.loads(line)
		tempDict = tempList['0']
		if(verticalName in tempDict.keys()):
			tempOutputFile = open(outputDir+'/'+tempDict[verticalName], 'a')
			json.dump(tempList, tempOutputFile)
			tempOutputFile.write('\n')
			tempOutputFile.close()
		else:
			noOfNoVertical += 1

	print('noOfNoVertical = ', noOfNoVertical)
	inputFile.close()

def new_verticalsFileSorting(inputFilename, outputFilename, fileWrite):
	#sort mobile > ../output.txt
	#inputFilename = '../new_verticals/'+inputFilename
	#outputFilename = '../sorted_new_verticals/'+outputFilename
	tempOutputFilename = '../output'
	command = 'sort '+inputFilename + ' -o '+ tempOutputFilename
	print(command.split(' '))
	subprocess.call(command.split(' '))#, shell = 'True')
	tempOutputFile = open(tempOutputFilename, 'r')	
	if fileWrite:
		outputFile = open(outputFilename, 'w')
		outputFile.close()
	outputFile = open(outputFilename, 'a')
	lineNo = 0
	for line in tempOutputFile:
		lineNo += 1
		if (lineNo != 1):
			outputFile.write(line)

	tempOutputFile.close()
	outputFile.close()

def new_handle_aiFba(inputDict):
	competitorName = 'competitor'
	aiFbaAttrName = 'ai_fba'

	allAttr = inputDict.keys()

	if not (competitorName in allAttr):
		return False

	if not (inputDict[competitorName] == 'ai'):
		return False

	aiFbaNumeric = {'false':-1, 'true':1}

	if (aiFbaAttrName in allAttr):
		inputDict[aiFbaAttrName] = aiFbaNumeric[inputDict[aiFbaAttrName]]
	else:
		inputDict[aiFbaAttrName] = 0.5
				
	return inputDict

def new_changeToNumeric(allAttrList, inputDict):
	allAttr = inputDict.keys()
	for attrName in allAttrList:
		if ( (not(attrName in allAttr)) or (inputDict[attrName] == 'null')):
			inputDict[attrName] = epsNull
		else:
			inputDict[attrName] = float(inputDict[attrName])

	return inputDict

def new_checkReqAttr(reqOneOfAttr, inputDict):
	reqOneOfAttrFlag = 0

	for i in range(len(reqOneOfAttr)):
		reqOneOfAttrFlag = 0
		for eachReqOneOfAttr in reqOneOfAttr[i]:
			if (abs(inputDict[eachReqOneOfAttr] - epsNull) > eps):
				reqOneOfAttrFlag = 1
				break
		if (reqOneOfAttrFlag == 0):
			break

	if (reqOneOfAttrFlag == 0):
		return False

	return True

def new_verticalsFileNormalization(inputFilename, outputFilename, fileWrite):
	#inputFilename = '../sorted_new_verticals/'+inputFilename
	#outputFilename = '../new_normalized_sorted_verticals/'+outputFilename

	inputFile = open(inputFilename, 'r')
	
	if fileWrite:
		outputFile = open(outputFilename, 'w')
		outputFile.close()

	outputFile = open(outputFilename, 'a')

	noOfLines = noOfCompetitors = 0
	timeStampIdx = 1
	allAttr = 0
	for line in inputFile:
		allAttrValueList = json.loads(line)
		inputDict = allAttrValueList['0']
		inputDict[timeStampName] = allAttrValueList['1'][1]
		noOfLines += 1
		tempAllAttr = inputDict.keys()

		#handle aiFba starts
		inputDict = new_handle_aiFba(inputDict)
		if not inputDict:
			continue
		#handle aiFba ends

		#All the values must be null or numeric or numeric string or must not be present
		#Changing all values to numeric values .. and after this step all attr will be present 
		inputDict = new_changeToNumeric(allAttrList, inputDict)

		allAttr = inputDict.keys()

		if not new_checkReqAttr(reqOneOfAttr, inputDict):
			continue

		#normalization
		for i in range(len(reqOneOfAttr)):
			allAvg = allSum = allCnt = 0.0
			for eachReqOneOfAttr in reqOneOfAttr[i]:
				if (abs(inputDict[eachReqOneOfAttr] - epsNull) > eps):
					allSum += inputDict[eachReqOneOfAttr]
					allCnt += 1
			allAvg = allSum/allCnt
			for eachReqOneOfAttr in reqOneOfAttr[i]:
				if (not (abs(inputDict[eachReqOneOfAttr] - epsNull) > eps)):
					inputDict[eachReqOneOfAttr] = allAvg

		json.dump(inputDict, outputFile)
		outputFile.write('\n')

		noOfCompetitors += 1


	inputFile.close()
	outputFile.close()
	print('noOfLines = ', noOfLines, '   noOfCompetitors = ', noOfCompetitors)
	

def new_makeTrainData(inputFilename, outputFilename, fileWrite):
	#inputFilename = '../new_normalized_sorted_verticals/' + inputFilename
	#outputFilename = '../new_trainData/' + outputFilename

	inputFile = open(inputFilename, 'r')
	
	if fileWrite:
		outputFile = open(outputFilename, 'w')
		outputFile.close()
	
	outputFile = open(outputFilename, 'a')

	priceRange = [[10000, 20000], [20000, 30000], [30000, 40000]]

	noOfaiAvgSla_LessThanEps = noOfTrain = pre_aiADSP = noOfFsn = 0
	preFsn = preTimeStamp = '0'

	tempCnt = zero
	noOfRange = len(priceRange)
	rangeCnt = [zero] * noOfRange

	for line in inputFile:
		inputDict = json.loads(line)
		fsn_Value = inputDict[fsnName] 
		timeStamp_Value = inputDict[timeStampName]
		aiADSP_Value = inputDict[aiADSP]
		fkSP_Value = inputDict[fkSP]
		mrpName_Value = inputDict[mrpName]
		aiFbaAttr_Value = inputDict[aiFbaAttrName]

		aiMinSla_Value = inputDict[aiMinSlaName]
		aiMaxSla_Value = inputDict[aiMaxSlaName]

		#try:
		tempList = []			

		aiADSP_diff_fkSP_Value = (aiADSP_Value - fkSP_Value)/mrpName_Value#fkSP_Value
		aiADSP_diff_mrpName_Value = (aiADSP_Value - mrpName_Value)/mrpName_Value
		aiAvgSla_Value = (aiMinSla_Value + aiMaxSla_Value)/2
		if(abs(aiAvgSla_Value - 0) <= eps):
			noOfaiAvgSla_LessThanEps += 1

		tempList.append(aiADSP_diff_fkSP_Value)
		tempList.append(aiAvgSla_Value)#aiFbaAttr_Value)#aiADSP_diff_mrpName_Value)
		tempList.append(aiFbaAttr_Value)
		if (preFsn == fsn_Value):
			aiADSP_diff = (aiADSP_Value - pre_aiADSP)
			aiADSP_diff_percent = aiADSP_diff/pre_aiADSP
		else:
			noOfFsn += 1
			aiADSP_diff = zero
			aiADSP_diff_percent = zero

		tempList.append(aiADSP_diff)
		tempList.append(aiADSP_diff_percent)

		tempList.append(aiADSP_Value)
		tempList.append(fkSP_Value)
		tempList.append(mrpName_Value)
		tempList.append(fsn_Value)
		tempList.append(timeStamp_Value)

		rangeNo = zero
		for rangeIdx in range(noOfRange):
			if ((mrpName_Value >= priceRange[rangeIdx][0]) and (mrpName_Value < priceRange[rangeIdx][1])):
				rangeNo = rangeIdx
				rangeCnt[rangeIdx] += 1
				tempCnt += 1
				break

		tempList.append(rangeNo)

		#train_aiADSP_diff_fkSP_List.append(aiADSP_diff_fkSP_Value)
		#train_aiADSP_diff_mrpName_List.append(aiADSP_diff_mrpName_Value)

		#trainValueList.append(tempList)
		json.dump(tempList, outputFile)
		outputFile.write('\n')
		noOfTrain += 1

		preFsn = fsn_Value
		preTimeStamp = timeStamp_Value
		pre_aiADSP = aiADSP_Value
		'''except:
			print(inputDict)
			
			print('----------------------------------------')
			for i in range(len(reqOneOfAttr)):
				for eachReqOneOfAttr in reqOneOfAttr[i]:
					print(eachReqOneOfAttr)
					print(inputDict[eachReqOneOfAttr])
					print( (abs(inputDict[eachReqOneOfAttr] - 0) > eps))
			print(fkSP_Value)
			print(mrpName_Value)
			for eachAttr in allAttrList:
				print(eachAttr)
				print(inputDict[eachAttr])
			
			sys.exit()
		'''
		

	inputFile.close()
	outputFile.close()

	print('tempCnt = ', tempCnt)

	for rangeIdx in range(noOfRange):
		print('rangeIdx = ', rangeIdx, '     rangeCnt = ', rangeCnt[rangeIdx])

	print('noOfTrain = ', noOfTrain, '   noOfFsn = ', noOfFsn)

def new_featureScalingTrainData(inputFilename, outputFilename, fileWrite):

	inputFile = open(inputFilename, 'r')
	
	if fileWrite:
		outputFile = open(outputFilename, 'w')
		outputFile.close()
	
	outputFile = open(outputFilename, 'a')
	
	scalableAttrIdxList = [0, 1, 2, 3, 4, 5, 6, 7, 9]
	max_scalableAttr = []
	for idx in range(scalableAttrIdxList[len(scalableAttrIdxList)-1] + 1):
		max_scalableAttr.append(zero)

	lineNo = 0

	nullAttr = 0

	for line in inputFile:
		lineNo += 1
		inputDict = json.loads(line)
		for idx in scalableAttrIdxList:
			if (lineNo == 1):
				max_scalableAttr[idx] = abs(inputDict[idx])
			else:
				max_scalableAttr[idx] = max(max_scalableAttr[idx], abs(inputDict[idx]))

	inputFile.close()

	inputFile = open(inputFilename, 'r')

	for line in inputFile:
		inputDict = json.loads(line)
		for idx in scalableAttrIdxList:
			if (abs(max_scalableAttr[idx] - zero) > eps):
				inputDict[idx] /= max_scalableAttr[idx]
		json.dump(inputDict, outputFile)
		outputFile.write('\n')

	inputFile.close()
	outputFile.close()
	#print ('nullAttr = ', nullAttr)

def new_plotData(valueListFilename):
	valueListFile = open(valueListFilename, 'r')

	lineNo = 0

	xAxis = 0 #7
	yAxis = 5

	for line in valueListFile:
		lineNo += 1
		
		'''if (lineNo > 1000):
			break
		'''

		#if ((lineNo > 900) and (lineNo < 1000)):
		trainValue = json.loads(line)
		for i in [5]:
			plt.plot(trainValue[xAxis], trainValue[i], colors[i], markersize = 10)

	plt.xlabel(trainAttrList[xAxis])
	plt.ylabel(trainAttrList[yAxis])

	print('line No = ', lineNo)
	plt.show()

def new_makeClusterKMeans(valueListFilename):
	for noOfClusters in [3]:
		noOfClusters += 1
		valueListFile = open(valueListFilename, 'r')

		partailFitTrainSize = 100
		#noOfClusters = 3
		lineNo = 0

		kmeans = KMeans(n_clusters = noOfClusters, max_iter=100)#MiniBatchKMeans(n_clusters = noOfClusters, max_iter=100, batch_size=100)

		trainValueList = []
		for line in valueListFile:
			lineNo += 1
			#if (lineNo % partailFitTrainSize == 0):
			'''	trainValueList = np.array(trainValueList)
				kmeans.partial_fit(trainValueList)
				trainValueList = []
			'''
			valueList = json.loads(line)
			tempList = []
			for col in clusterAttrList:
				tempList.append(valueList[col])
			trainValueList.append(tempList)
		
		#if (lineNo % partailFitTrainSize != 0):
		trainValueList = np.array(trainValueList)
		#kmeans.partial_fit(trainValueList)
		kmeans.fit(trainValueList)

		valueListFile.close()

		centroids = kmeans.cluster_centers_
		labels = kmeans.labels_
		
		print('centroids = ', centroids)
		print('labels = ', len(labels))

		valueListFile = open(valueListFilename, 'r')
		
		ansList = []

		xAxis = 0
		yAxis = 5

		scoreSum = 0
		lineNo = 0
		for line in valueListFile:
			valueList = json.loads(line)
			'''
			tempList = []
			for col in clusterAttrList:
				tempList.append(valueList[col])
			'''
			#ansList.append(kmeans.predict([tempList])[0])
			#thisLabel = kmeans.predict([tempList])[0]
			plt.plot(valueList[xAxis], valueList[yAxis], colors[labels[lineNo]], markersize = 10)
			#scoreSum += kmeans.score([tempList])
			lineNo += 1
		
		scoreSum = kmeans.score(trainValueList)
		
		plt.xlabel(trainAttrList[xAxis])
		plt.ylabel(trainAttrList[yAxis])

		#plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'x', s = 150, linewidths = 5, zorder = 10)

		plt.show()
		

		valueListFile.close()
		print('noOfClusters = ', noOfClusters, '      sum = ', scoreSum)
		#print (len(ansList))
		#print(ansList[1])
# new ends


def main():
	#cat('test.txt')
	#regex()
	#inputDumpFileProcessing('../blah10power7', '../sampleData.txt')
	#processOnEachVertical()
	#makeClusterKMeans('../trainValueListFile')
	fsnList = ['MOBCP7K5F5GZYCZF', 'MOBCPHUBZ6A4W7BD', 'MOBCPHUF5EYDP2GB', 'MOBCPHUFX4ZWYRH8', 'MOBCPR9Y2GPXEHUX', 'MOBCPUQQFAEEVSU9', 'MOBCPUXBYKRVQJT4', 'MOBCQHXTKQ2NTNF2', 'MOBCR23AKZSRGQSU', 
	'MOBCR43GQJHQ9HDX', 'MOBCRRPGERT5FZCW', 'MOBCTEKWTP9PF2ST', 'MOBCTF99CSDYZEMU', 'MOBCU43EFESAZKE2', 
	'MOBCV849GZNHTK2F']
	
	'''
	lot1File = open('../lot1fsnList', 'r')
	fsnList = []
	for line in lot1File:
		line = re.search(r'\S+', line).group()
		fsnList.append(line)
	#print(len(fsnList))
	#print(fsnList)
	lot1File.close()
	'''
	'''
	#for eachFsn in fsnList:
	new_inputDumpFile_to_cleaned('../fsnDumpData/MOBCPHUF5EYDP2GB', '../fsnDumpData/cleanedInputDumpFile', 'a')
	
	new_cleaned_to_fsnInputFile('../fsnDumpData/cleanedInputDumpFile', '../fsnDumpData/fsnInputFile', 'a', 2)
	
	new_verticalClassifier('../fsnDumpData/fsnInputFile', '../fsnDumpData/new_verticals')
	
	new_verticalsFileSorting('../fsnDumpData/new_verticals/mobile', '../fsnDumpData/sorted_new_verticals/mobile', True)
	
	new_verticalsFileNormalization('../fsnDumpData/sorted_new_verticals/mobile', '../fsnDumpData/new_normalized_sorted_verticals/mobile', True)
	
	new_makeTrainData('../fsnDumpData/new_normalized_sorted_verticals/mobile', '../fsnDumpData/new_trainData/mobile', True)
	
	new_featureScalingTrainData('../fsnDumpData/new_trainData/mobile', '../fsnDumpData/new_featureScaledTrainData/mobile', True)
	
	new_plotData('../fsnDumpData/new_featureScaledTrainData/mobile')
	'''
	new_makeClusterKMeans('../fsnDumpData/new_featureScaledTrainData/mobile')
	'''
	print (eps)
	'''

if __name__ == '__main__':
	main()

