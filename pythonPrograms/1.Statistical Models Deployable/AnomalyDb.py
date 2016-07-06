from library import *
from skylineConstants import *

class AnomalyDb(object):

	configKey = 'AnomalyDb'	

	def __init__(self, configDict):
		configDict = configDict[self.configKey]
		self.ip = configDict['ip']
		self.userName = configDict['userName']
		self.password = configDict['password']
		self.dbName = configDict['dbName']
		self.tableName = configDict['tableName']

	def dbConnect(self):
		ip = self.ip
		userName = self.userName
		password = self.password
		dbName = self.dbName
		tableName = self.tableName

		# Open database connection
		dbConnectObj = MySQLdb.connect(host = ip, user = userName, passwd = password,db = dbName)#,port = 3306)
		# prepare a cursor object using cursor() method
		cursorObj = dbConnectObj.cursor()

		self.dbConnectObj = dbConnectObj
		self.cursorObj = cursorObj

	def insert(self, insertDict):
		tableName = self.tableName
		dbConnectObj = self.dbConnectObj
		cursorObj = self.cursorObj	

		query = 'INSERT INTO '+tableName+'('
		lenColNameList = len(prVerTbColNameList)
		
		colNo = 0
		for eachCol in prVerTbColNameList:
			colNo += 1
			if (colNo < lenColNameList):
				query += eachCol+', '
			else:
				query += eachCol+') '

		query += 'VALUES ('
		colNo = 0
		for eachCol in prVerTbColNameList:
			colNo += 1
			if (colNo < lenColNameList):
				query += '"'+str(insertDict[eachCol]) + '", '
			else:
				query += '"'+str(insertDict[eachCol]) + '")'

		try:
			#print('query = ', query)
			cursorObj.execute(query)
			dbConnectObj.commit()
		except:
			print('Could not execute insert query')
			dbConnectObj.rollback()

	def checkIfTuplePresent(self, checkDict):
		tableName = self.tableName
		dbConnectObj = self.dbConnectObj
		cursorObj = self.cursorObj	

		query = 'SELECT EXISTS(SELECT 1 FROM ' + tableName + ' WHERE '
		lenColNameList = len(prVerTbColNameList)
		
		colNo = 0
		for eachCol in prVerTbColNameList:
			colNo += 1
			if (colNo < lenColNameList):
				query += eachCol+' = "' + str(checkDict[eachCol]) + '" AND '
			else:
				query += eachCol+' = "' + str(checkDict[eachCol]) + '") '

		isPresent = False

		try:
			#print('query = ', query)
			cursorObj.execute(query)
			isPresent = cursorObj.fetchone()[0]
			#print('isPresent = ', isPresent)
			dbConnectObj.commit()
		except:
			print('Could not execute checkIfTuplePresent query')
			dbConnectObj.rollback()

		return isPresent

	def selection(self):
		tableName = self.tableName
		dbConnectObj = self.dbConnectObj
		cursorObj = self.cursorObj	

		query = 'SELECT * FROM ' + tableName

		try:
			#print('query = ', query)
			#print('ans = ')
			cursorObj.execute(query)
			output = cursorObj.fetchall()
			print(output)
			dbConnectObj.commit()
		except:
			print('Could not execute select query')
			dbConnectObj.rollback()

	def deleteAllTuples(self):
		tableName = self.tableName
		dbConnectObj = self.dbConnectObj
		cursorObj = self.cursorObj	

		query = 'DELETE FROM '+tableName

		try:
			#print('query = ', query)
			#print('ans = ')
			cursorObj.execute(query)
			dbConnectObj.commit()
		except:
			print('Could not execute select query')
			dbConnectObj.rollback()

	def dbClose(self):
		dbConnectObj = self.dbConnectObj
		dbConnectObj.close()
'''
def main():
	configFilename = '../Config5/checkForAnomaly.yml'
	with open(configFilename, 'r') as configFile:
		configDict = yaml.load(configFile)

	anomalyDbObj = AnomalyDb(configDict)
	anomalyDbObj.dbConnect()
	#insertDict = {timeStampName:1466056120312.0, fsnName:'ACCEE9M6BSAJEF2P'}
	#anomalyDbObj.insert(insertDict)
	#isPresent = anomalyDbObj.checkIfTuplePresent(insertDict)
	anomalyDbObj.selection()
	#anomalyDbObj.deleteAllTuples()
	#print('isPresent', isPresent)

	print('db Connected')

if __name__ == '__main__':
	main()
'''
