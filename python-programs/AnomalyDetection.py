from library import *
from skylineConstants import *

from StddevFromMovingAverage import StddevFromMovingAverage
from HistogramBins import HistogramBins


#skyline starts

class AnomalyDetection(object):

	def new_checkForAnomalyOneAttr(self, checkList, NetAlgos, configDict):

		flag = False

		flagScore = 0
		algoWt = configDict['Skyline']

		ret = {}

		stddevFromMovingAverageObj = StddevFromMovingAverage(configDict)
		(retFlag, retStr, retDict) = stddevFromMovingAverageObj.solve(checkList)
		flag = flag or retFlag
		flagScore += algoWt['StddevFromMovingAverage'] * int(retFlag)
		ret.update(retDict)

		histogramBinsObj = HistogramBins(configDict)
		'''
		try:
			checkList = (checkList - checkList.shift())#/checkList
		except:
			pass
		checkList.dropna(inplace = True)
		'''
		(retFlag, retStr, retDict) = histogramBinsObj.solve(checkList)
		flag = flag or retFlag
		flagScore += algoWt['HistogramBins'] * int(retFlag)
		ret.update(retDict)

		if NetAlgos:
			flag = (flagScore - algoWt['AlgosThreshold'] >= 0)

		return (flag, ret)


	def new_checkForAnomaly(self, checkList, configDict):
		minNoData = configDict['BasicConfig']['minNoData']
		retFlag = False
		retDict = {}
		if (len(checkList) >= minNoData):
			NetAlgos = True
			retFlag = True
			for eachAnomalyAttr in anomalyAttrList:
				eachAttrCheckList = checkList.loc[:, eachAnomalyAttr]
				(tempFlag, tempDict) = self.new_checkForAnomalyOneAttr(eachAttrCheckList, NetAlgos, configDict)
				retFlag = retFlag and tempFlag
				retDict.update(tempDict)

		return (retFlag, retDict)
