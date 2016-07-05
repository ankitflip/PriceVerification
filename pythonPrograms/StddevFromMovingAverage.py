from library import *
from skylineConstants import *

from SeriesStatFunc import SeriesStatFunc

class StddevFromMovingAverage(object):

	configKey = 'StddevFromMovingAverage'

	def __init__(self, configDict):
		configDict = configDict[self.configKey]
		self.com = configDict['com']
		self.maxTimesThanStdDev = configDict['maxTimesThanStdDev']
		self.minTimesThanMean = configDict['minTimesThanMean']
		self.minBaseDifference = configDict['minBaseDifference']

	def solve(self, timeseries):
		"""
		A timeseries is anomalous if the absolute value of the average of the latest
		datapoint minus the moving average is greater than three standard
		deviations of the moving average. This is better for finding anomalies with
		respect to the short term trends.
		"""

		com = self.com
		maxTimesThanStdDev = self.maxTimesThanStdDev
		minTimesThanMean = self.minTimesThanMean
		minBaseDifference = self.minBaseDifference
		seriesStatFuncObj = SeriesStatFunc(timeseries)
		series = pd.Series([x for x in timeseries])
		expAverage = series.ewm(com).mean()
		stdDev = series.ewm(com).std()
		if not (seriesStatFuncObj.minConditionFromMean(abs(series.iloc[-1] - expAverage.iloc[-1]), minTimesThanMean, minBaseDifference)):
			retFlag = False
		else:
			retFlag = abs(series.iloc[-1] - expAverage.iloc[-1]) > maxTimesThanStdDev * stdDev.iloc[-1]
		'''
		print('series = ')
		print(series)
		print('expAvg = ')
		print(expAverage)
		print('stdDev = ')
		print(stdDev)
		sys.exit()
		'''
		retStr = 'StddevFromMovingAverage :- '+str(retFlag)+' : '
		retStr += 't-expAvg = '+ str(abs(series.iloc[-1] - expAverage.iloc[-1]))+' : stdDev = '+str(stdDev.iloc[-1])

		retDict = {SFMAvgFlagName : retFlag, tailMinusExpAvgName : abs(series.iloc[-1] - expAverage.iloc[-1]), stdDevName : (stdDev.iloc[-1])}

		return (retFlag, retStr, retDict)