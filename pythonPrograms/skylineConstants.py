
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

zero = 0

comName = "competitor"
fsnName = 'fsn'
timeStampName = 'timestamp'

comAi = "AI"
aiFbaAttrName = 'ai_fba'
aiLSCWFName = 'ai_lowest_shipping_charge_without_fba'
aiMrpName = 'ai_mrp'
mrpName = 'mrp'
aiADSP = 'ai_displayed_selling_price'
aiDP = 'ai_displayed_price'
fkSP = 'flipkart_selling_price'
fkDiff = 'competitor_flipkart_diff'
verAttrName = 'vertical'

shippingDaysName = 'shipping_days'
aiMinSlaName = 'ai_min_sla'
aiMaxSlaName = 'ai_max_sla'

anomalyAttrList = [aiDP]#, fkDiff]


aiColNoStart = 4
aiPriceCol = range(4,10)
#print(aiPriceCol)

colors = ['g.', 'r.', 'c.', 'b', 'g.', 'r.', 'c.', 'b.']

tail = 3
extendedTail = 5*tail

oneDay = 86400000
#noOfDays = 60

#minNoData = 20

algoName = 'algoName'
anomalyAttrName = 'isAnomaly'
noOfDataAttrName = 'noOfData'

#StddevFromMovingAverage
tailMinusExpAvgName = 'tail-expAvg'
stdDevName = 'stdDev'
SFMAvgFlagName = 'SFMAvgFlag'

#HistogramBins
tailName = 'tail'
binSizeName = 'binSize'
HBinsFlagName = 'HBinsFlag'
minBinSizeName = 'minBinSize'

resultColNameList = [comName, verAttrName, fsnName, aiDP, noOfDataAttrName, timeStampName, anomalyAttrName, SFMAvgFlagName, tailMinusExpAvgName, stdDevName, HBinsFlagName, tailName, binSizeName, minBinSizeName]

#priceVerificationTable
prVerTbColNameList = [timeStampName, fsnName]

#exceptionFile
notifName = 'Kafka_Notif'
exceptionName = 'exception'
exceptionColNameList = [notifName, exceptionName]