#from collections import Counter
import sys, re, json
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
from sets import Set
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import MySQLdb
import subprocess
import math
import time
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

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


allAttrList = [timeStampName, aiADSP, aiLSP, aiLP, aiDP]
reqOneOfAttr = [[timeStampName], [aiADSP, aiLSP, aiLP, aiDP]]


aiColNoStart = 4
aiPriceCol = range(4,10)
#print(aiPriceCol)

colNameList = ['fsn', 'timestamp', 'shipping_days', 
'flipkart_selling_price', 'ai_mrp', 'ai_lowest_price', 
'ai_lowest_selling_price', 'ai_lowest_shipping_charge', 'ai_displayed_price', 
'ai_displayed_selling_price', 'ai_displayed_shipping_charge', 'ai_promo', 
'ai_promo_percent', 'ai_min_sla', 'ai_max_sla', 
'ai_min_sla_without_fba', 'ai_max_sla_without_fba', 'ai_stock_status', 
'ai_crawl_status', 'ai_lowest_priced_seller', 'ai_bread_crumb', 
'ai_displayed_seller', 'ai_fba', 'ai_lowest_seller_fba', 
'ai_lowest_seller_promo', 'ai_lowest_seller_promo_percent', 'ai_lowest_price_without_fba', 
'ai_lowest_selling_price_without_fba', 'ai_lowest_shipping_charge_without_fba', 'ai_lowest_priced_seller_without_fba', 
'ai_lowest_seller_without_fba_promo', 'ai_lowest_seller_without_fba_promo_percent', 'ai_lowest_price_with_fba', 
'ai_lowest_selling_price_with_fba', 'ai_lowest_shipping_charge_with_fba', 'ai_lowest_priced_seller_with_fba', 
'ai_lowest_seller_with_fba_promo', 'ai_lowest_seller_with_fba_promo_percent', 'ai_translate_status']


colors = ['g.', 'r.', 'c.', 'b', 'g.', 'r.', 'c.', 'b.']

#skyline starts

tail = 3
extendedTail = 5*tail

rolWindow = 10
hundred = 100

def changeTimeStamp(timeStamp):
	thousand = 1000
	timeStamp /= thousand
	retTimeStamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timeStamp))
	return retTimeStamp

def test_stationarity(timeseries):
	#timeseries = np.log(timeseries)
	#Determing rolling statistics
	rolmean = timeseries.rolling(window = rolWindow).mean() #pd.rolling_mean(timeseries, window=12)
	rolstd = timeseries.rolling(window = rolWindow).std()

	'''
	#Plot rolling statistics:
	orig = timeseries.plot(color='blue',label='Original')
	mean = rolmean.plot(color='red', label='Rolling Mean')
	std = rolstd.plot(color='black', label = 'Rolling Std')
	plt.legend(loc='best')
	plt.title('Rolling Mean & Standard Deviation')
	plt.show()
	'''

	#Perform Dickey-Fuller test:
	
	print 'Results of Dickey-Fuller Test:'
	dftest = adfuller(timeseries, autolag='AIC')
	dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
	for key,value in dftest[4].items():
		dfoutput['Critical Value (%s)'%key] = value
	print dfoutput
	
def ARIMA_Model(inputList):
	pdFrame = pd.DataFrame(inputList)
	
	pdFrame = pdFrame.loc[:, [timeStampName, aiDP]]

	pdFrame[timeStampName] = pdFrame[timeStampName].apply(changeTimeStamp)
	print('\n--------- isinstance -----------------')
	print(isinstance(pdFrame[timeStampName], pd.DatetimeIndex))
	print('\n---------- col instance done----------------')

	pdFrame = pdFrame.set_index(timeStampName)
	pdFrame = pdFrame.sort_index(ascending = True)
	pdFrame = pdFrame[aiDP]
	pdFrame.index = pd.to_datetime(pdFrame.index)
	print(pdFrame.head())
	print(isinstance(pdFrame.index, pd.DatetimeIndex))
	#sys.exit()

	'''
	rolmean = pdFrame.rolling(window = rolWindow).mean()
	pdFrame_rolmean_diff = pdFrame - rolmean
	#print(pdFrame_rolmean_diff.head(rolWindow))
	pdFrame_rolmean_diff.dropna(inplace = True)
	#print(pdFrame_rolmean_diff.head(rolWindow))

	test_stationarity(pdFrame_rolmean_diff)

	
	print('\n ------------------ pdFrame_pdFrameEwma_diff ---------------------')
	pdFrameEwma = pdFrame.ewm(halflife = rolWindow).mean()
	pdFrame_pdFrameEwma_diff = pdFrame - pdFrameEwma
	print(pdFrame_pdFrameEwma_diff.head())
	test_stationarity(pdFrame_pdFrameEwma_diff)
	'''

	print('\n------------------ pdFrame_diff ----------------------------------')
	pdFrame_diff = pdFrame - pdFrame.shift()
	print(pdFrame_diff.head())
	pdFrame_diff.dropna(inplace = True)
	lag_acf = acf(pdFrame_diff, nlags=rolWindow)
	#Plot ACF:
	'''
	plt.subplot(121) 
	plt.plot(lag_acf)
	plt.axhline(y=0,linestyle='--',color='black')
	plt.axhline(y=-1.96/np.sqrt(len(pdFrame_diff)),linestyle='--',color='black')
	plt.axhline(y=1.96/np.sqrt(len(pdFrame_diff)),linestyle='--',color='black')
	plt.title('Autocorrelation Function')
	

	lag_pacf = pacf(pdFrame_diff, nlags=rolWindow, method='ols')
	#Plot PACF:
	
	plt.subplot(122)
	plt.plot(lag_pacf)
	plt.axhline(y=0,linestyle='--',color='black')
	plt.axhline(y=-1.96/np.sqrt(len(pdFrame_diff)),linestyle='--',color='black')
	plt.axhline(y=1.96/np.sqrt(len(pdFrame_diff)),linestyle='--',color='black')
	plt.title('Partial Autocorrelation Function')
	plt.tight_layout()

	plt.show()
	'''
	#sys.exit()
	p = 2
	q = 0
	d = 0
	model = ARIMA(pdFrame_diff, order=(p, d, q))  
	results_ARIMA = model.fit(disp=-1)  
	
	plt.plot(pdFrame_diff)
	plt.plot(results_ARIMA.fittedvalues, color='red')
	plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-pdFrame_diff)**2))
	plt.show()
	
	test_stationarity(pdFrame_diff)

	predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
	predictions_ARIMA = [pdFrame.iloc[0]]

	for i in range(len(predictions_ARIMA_diff)):
		predictions_ARIMA.append(predictions_ARIMA_diff.iloc[i]+pdFrame.iloc[i])
	
	plt.plot(pdFrame)
	plt.plot(predictions_ARIMA, color = 'red')
	plt.show()
	
	

	'''
	mavg = pdFrame.rolling(window=2, min_periods=1).mean()
	print(mavg.tail())
	pdFrame.plot(label = aiDP)
	mavg.plot(label = 'mavg')
	plt.legend()
	plt.show()
	'''
def new_checkForAnomaly(inputFilename):
	
	inputFile = open(inputFilename, 'r');

	inputList = []

	lineNo = zero

	lastLineNo = 200#109
	startLineNo = 0

	for line in inputFile:
		lineNo += 1
		if(lineNo > lastLineNo):
			break
		if ((lineNo>=startLineNo) and (lineNo<=lastLineNo)):
			inputDict = json.loads(line)
			inputList.append(inputDict)

	inputFile.close()

	ARIMA_Model(inputList)

	#sys.exit()
	

#timeSeries ends

def main():
	new_checkForAnomaly('../fsnDumpData/skyLine/trainData')

if __name__ == '__main__':
	main()