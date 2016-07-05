from library import *

class SeriesStatFunc(object):

	def __init__(self, timeseries):
		series = pd.Series([x for x in timeseries])
		self.series = series

	def calMean(self):
		return self.series.mean()

	def calStdDev(self):
		return self.series.std()

	def calMedian(self):
		return self.series.median()

	def calPercentile(self, percent):
		return np.percentile(self.series, percent)

	def getStatValue(self, choice, value):
		if (choice == 'mean'):
			return self.calMean()
		elif (choice == 'stdDev'):
			return self.calStdDev()
		elif (choice == 'median'):
			return self.calMedian()
		elif (choice == 'percent'):
			return self.calPercentile(value)

	def minConditionFromMean(self, testValue, percent, minBaseDifference):
		return (testValue >= abs(percent*self.series.mean()) + abs(minBaseDifference))