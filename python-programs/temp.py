from library import *
from skylineConstants import *

def calAccuracy(inputFilename, outputFilename, configFilename, outputFileWrite):
	with open(configFilename, 'r') as configFile:
		configDict = yaml.load(configFile)

	minNoData = configDict['BasicConfig']['minNoData']
	ewmThreshold = configDict['StddevFromMovingAverage']['maxTimesThanStdDev']

	inputDF = pd.read_csv(inputFilename, index_col=False, header=0)

	confidenceLevelName = 'confidenceLevel'

	if (outputFileWrite):
		outputFile = open(outputFilename, 'w')
		wr = csv.writer(outputFile, quoting=csv.QUOTE_ALL)
		wr.writerow(resultColNameList + [confidenceLevelName])

		outputFile.close()

	outputFile = open(outputFilename, 'a')
	wr = csv.writer(outputFile, quoting=csv.QUOTE_ALL)

	for idx, eachRowInputDF in inputDF.iterrows():
		print("idx = ", idx)
		try:
			resultList = []

			if (eachRowInputDF[noOfDataAttrName] >= minNoData):
				for eachCol in resultColNameList:
					resultList.append(eachRowInputDF[eachCol])

				confidenceEwm = (eachRowInputDF[tailMinusExpAvgName]*1.0 - ewmThreshold*eachRowInputDF[stdDevName])/(eachRowInputDF[aiDP] + 1)
				confidenceHist = (eachRowInputDF[minBinSizeName]*1.0 - eachRowInputDF[binSizeName])/(eachRowInputDF[minBinSizeName] + 1)
				sign = 1
				if ( (confidenceEwm < 0) or (confidenceHist < 0) ):
					sign = -1
				confidenceLevelValue = abs(confidenceEwm * confidenceHist) * 100 * sign 
				if (not eachRowInputDF[anomalyAttrName]):
					confidenceLevelValue = None
				resultList.append(confidenceLevelValue)
				wr.writerow(resultList)
		except Exception as e:
			print("Exception = ")
			print(e)

	outputFile.close()

def main():
	configFolder = '../Config1.5-5/'
	inputFilename = configFolder+'aiDPHistoryResultHistChanged.csv'
	outputFilename = configFolder+'aiDPHistoryResultHistChangedConf.csv'
	configFilename = configFolder+'checkForAnomaly.yml'

	calAccuracy(inputFilename, outputFilename, configFilename, True)

if __name__ == '__main__':
	main()

