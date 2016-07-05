from library import *
from skylineConstants import *

class HistogramBins(object):

	configKey = 'HistogramBins'

	def __init__(self, configDict):
		configDict = configDict[self.configKey]
		self.rangeBinSize = configDict['rangeBinSize']
		self.noOfBins = configDict['noOfBins']
		self.minBinSize = configDict['minBinSize']

	def solve(self, timeseries):
		"""
		A timeseries is anomalous if the average of the last three datapoints falls
		into a histogram bin with less than 20 other datapoints (you'll need to tweak
		that number depending on your data)
		Returns: the size of the bin which contains the tail_avg. Smaller bin size
		means more anomalous.
		"""
		series = pd.Series([x for x in timeseries])
		rangeBinSize = int( (((self.rangeBinSize*1.0)/100) * min(series)) ) + 1
		noOfBins = int( ((max(series) - min(series))/rangeBinSize)) + 1
		noOfData = len(series)
		minBinSize = int( ((self.minBinSize*1.0)/100) * noOfData) + 1
		t = series.iloc[-1]
		h = np.histogram(series, bins=noOfBins)
		bins = h[1]

		print('rangeBinSize = %d  noOfBins = %d   max-min= %d minBinSize = %d' %(rangeBinSize, noOfBins, (max(series) - min(series)), minBinSize) )

		retFlag = False
		for index, bin_size in enumerate(h[0]):
			print('bin = ', (bins[index], bins[index+1]))
			if bin_size <= minBinSize:
				# Is it in the current bin?
				if t >= bins[index] and t < bins[index + 1]:
					retFlag = True
					break
			elif t >= bins[index] and t < bins[index + 1]:
				retFlag = False
				break

		if ( (t>=bins[index] and t<=bins[index+1]) and (bin_size<=minBinSize)):
			retFlag = True

		retStr = 'HistogramBins :- '+ str(retFlag)+' : '
		retStr += 't = '+str(t)+' : bin_size = ' + str(bin_size)
		retDict = {HBinsFlagName : retFlag, tailName : abs(t-series.iloc[-2]), binSizeName : bin_size, minBinSizeName : minBinSize}

		return (retFlag, retStr, retDict)