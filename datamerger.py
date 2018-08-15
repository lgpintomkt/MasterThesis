# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:30:54 2018

@author: LGPinto
"""

import featextractor as fe
import random
import pandas as pd
import csv
import os
from time import gmtime, strftime

class DataMerger():
    
    def __init__(self, churnPath, targetPath, geoDemographicPath, 
                 technographicPath, searchPath, appPath, 
                 clickstreamPath, socialGraphPath, samplePath, sampleRate = 0.004,
                 searchDict=None,demoDict=None,takenPkg=None,urlCodes=None,
                 cardTypeCodes=None,sourceCodes=None,packageNameCodes=None,
                 takenNodes=None,deviceDict=None,androidDict=None):
        
        self.searchDict = searchDict
        self.demoDict = demoDict
        self.takenPkg = takenPkg
        self.urlCodes = urlCodes
        self.cardTypeCodes = cardTypeCodes
        self.sourceCodes = sourceCodes
        self.packageNameCodes = packageNameCodes
        self.takenNodes = takenNodes
        self.deviceDict = deviceDict
        self.androidDict = androidDict
                
        self.paths=[('churn',churnPath,targetPath),  ('click',clickstreamPath),  
                    ('geo',geoDemographicPath), ('demo',geoDemographicPath), 
                    ('techno',technographicPath), ('search',searchPath),
                    ('app',appPath), ('social',socialGraphPath), 
                    ] #order is important
#        
#        self.paths=[('churn',churnPath,targetPath), ('app',appPath), ('search',searchPath),  ('click',clickstreamPath),  
#                    ('geo',geoDemographicPath), ('demo',geoDemographicPath), 
#                    ('techno',technographicPath), 
#                     ('social',socialGraphPath), 
#                    ] #order is important
        
        self.sampleRate = sampleRate
        self.samplePath = samplePath
    
    def _extractSample(self, path, sampleRate, resample, prevSample):
        #detect id column
        csvfile = open(path, 'rt', encoding='utf8')
        rowReader = csv.reader(csvfile, delimiter=',', quotechar='"')
        header = next(rowReader)
        csvfile.close()
        
        idColumn=0
        while(header[idColumn]!='id'):
            idColumn+=1
        
        userIds = set()
        with open(path, encoding='mbcs') as csvfile:
            rowReader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(rowReader) #skip header
            for row in rowReader:
                userIds.add(str(row[idColumn]))
        
        n = len(userIds)
        s = int(n*sampleRate)
        if(resample==False):    
            return random.sample(userIds,s)
        else:
            if(prevSample==None):
                raise ValueError('You need to set the sample variable.')
            if(len(prevSample)==n):
                raise GeneratorExit
            else:
                newSize = 0
                finalSample=set()
                gap=s
                while(newSize!=s):
                    finalSample.update(random.sample(userIds,gap))
                    newSize = len(finalSample)
                    gap = s-newSize
                return finalSample
    
    def _merge(self,path,sample,resample=False):
        
        if(path[0]=='churn'):
            print(' ')
            print('** Initiating Target Variable Feature Extraction **')
            print('Current Timestamp: '+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            churnData = fe.TargetFeatureExtractor(path[1], path[2], sample).features
            return pd.merge(churnData, self.timestampID, how='outer', on='id').to_sparse()
            
        elif(path[0]=='geo'):
            print(' ')
            print('** Initiating Geographic Feature Extraction **')
            print('Current Timestamp: '+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            self.categoricalIndex = len(self.mergedData.columns.values)+1+4
            return fe.GeoFeatureExtractor(path[1], sample).features
    
        elif(path[0]=='demo'):
            print(' ')
            print('** Initiating Demographic Feature Extraction **')
            print('Current Timestamp: '+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            self.demoDict, demoData = fe.DemoFeatureExtractor(path[1], sample, languageCodes=self.demoDict).features
            return demoData
            
        elif(path[0]=='techno'):
            print(' ')
            print('** Initiating Technographic Feature Extraction **')
            print('Current Timestamp: '+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            self.deviceDict, self.androidDict, technoData = fe.TechnoFeatureExtractor(path[1], sample, deviceCodes=self.deviceDict, androidVersionCodes=self.androidDict).features
            return technoData
            
        elif(path[0]=='search'):
            print(' ')
            print('** Initiating Search Query Feature Extraction **')
            print('Current Timestamp: '+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            folders=os.listdir(path[1])
            newSearches=pd.SparseDataFrame(columns=['id','searches'])
            newQueries=None
            for count,folder in enumerate(folders):
                print(' ')
                print('Extracting Search Data Batch '+str(count+1)+' out of '+str(len(folders)))
                print(' ')
                self.searchDict, searchData = fe.SearchQueryFeatureExtractor(path[1]+folder, sample, searchQueryDictionary=self.searchDict).features
                newSearches=pd.concat([newSearches,searchData[['id','searches']]])
                if(newQueries is None):
                    newQueries=searchData.drop('searches',1)
                else:
                    newQueries=pd.concat([newQueries,searchData.drop('searches',1)])
            if(len(folders)>1):
                newSearches=newSearches.groupby('id').agg('mean')
                newSearches['id']=newSearches.index
                newQueries['idTimestamp'] = newQueries[['id', 'timestamp']].apply(lambda x: ''.join(x.apply(str)), axis=1)
                newQueries=newQueries.drop(['id','timestamp'],1).groupby(['idTimestamp']).agg('max')
                newQueries['idTimestamp']=newQueries.index
                newQueries['timestamp'] = newQueries['idTimestamp'].apply(lambda x: x[-6:])
                newQueries['id'] = newQueries['idTimestamp'].apply(lambda x: x[:-6])
                newQueries=newQueries.drop('idTimestamp',1)
            searchData=pd.merge(newSearches,newQueries,on='id',how='outer').reset_index(drop=True).to_sparse()
            return searchData
        
        elif(path[0]=='app'):
            print(' ')
            print('** Initiating App Download Feature Extraction **')
            print('Current Timestamp: '+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            folders=os.listdir(path[1])
            newDownloads=pd.SparseDataFrame(columns=['id','downloads'])
            newApps=None
            for count,folder in enumerate(folders):
                print(' ')
                print('Extracting App Data Batch '+str(count+1)+' out of '+str(len(folders)))
                print(' ')
                self.takenPkg, appData = fe.AppDownloadFeatureExtractor(path[1]+folder, sample, takenPkg=self.takenPkg).features
                newDownloads=pd.concat([newDownloads,appData[['id','downloads']]])
                if(newApps is None):
                    newApps=appData.drop('downloads',1)
                else:
                    newApps=pd.concat([newApps,appData.drop('downloads',1)])
            if(len(folders)>1):
                newDownloads=newDownloads.groupby('id').agg('mean')
                newDownloads['id']=newDownloads.index
                newApps['idTimestamp'] = newApps[['id', 'timestamp']].apply(lambda x: ''.join(x.apply(str)), axis=1)
                newApps=newApps.drop(['id', 'timestamp'],1).groupby(['idTimestamp']).agg('max')
                newApps['idTimestamp']=newApps.index
                newApps['timestamp'] = newApps['idTimestamp'].apply(lambda x: x[-6:])
                newApps['id'] = newApps['idTimestamp'].apply(lambda x: x[:-6])
                newApps=newApps.drop('idTimestamp',1)
            appData=pd.merge(newDownloads,newApps,on='id',how='outer').reset_index(drop=True).to_sparse()
            return appData
        
        elif(path[0]=='click'):
            print(' ')
            print('** Initiating Clickstream Feature Extraction **')
            print('Current Timestamp: '+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            self.urlCodes, self.cardTypeCodes, self.sourceCodes, self.packageNameCodes, clickData = fe.ClickstreamFeatureExtractor(path[1], sample, urlCodes=self.urlCodes, cardTypeCodes=self.cardTypeCodes, sourceCodes=self.sourceCodes, packageNameCodes=self.packageNameCodes).features
            return clickData
            
        elif(path[0]=='social'):
            print(' ')
            print('** Initiating Social Graph Feature Extraction **')
            print('Current Timestamp: '+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            self.takenNodes, socialData = fe.SocialGraphFeatureExtractor(path[1], sample=sample, timestamps=self.days, takenNodes=self.takenNodes).features
            return socialData
        
    def merge(self,resample=False,prevSample=None):
        paths=self.paths
        samplePath=self.samplePath
        sampleRate=self.sampleRate
        
        #Id Sample
        print('** Extracting ID Sample from Path **')
        try:
            sample = self._extractSample(samplePath,sampleRate,resample,prevSample)
        except GeneratorExit:
            return pd.DataFrame()
        print('Final sample size: '+str(len(sample)))
        
        #Days
        month = '08'
        year = '17'
        days = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14',
                '15','16','17','18','19','20','21','22','23','24','25','26','27','28',
                '29','30']
        index = 0
        for day in days:
            days[index] = day+month+year
            index+=1
        
        print(' ')
        print('** Generating Timestamp ID Association **')
        timestampIdTuples = []
        for userId in sample:
            for day in days:
                timestampIdTuples.append((str(userId),str(day)))
        ids = [row[0] for row in timestampIdTuples]
        timestamps = [row[1] for row in timestampIdTuples]
        
        self.timestampID = pd.DataFrame({'id': ids,'timestamp': timestamps}).to_sparse()
        self.days=days
        
        firstFlag=1
        for path in paths:
            print(' ')
            newData = self._merge(path,sample)
            if('timestamp' in newData.columns.values):
                if(firstFlag==1):
                    self.mergedData=newData
                    firstFlag=0
                else:
                    self.mergedData = pd.merge(self.mergedData, newData, how='outer', on=['id','timestamp']).to_sparse()
            else:
                if(firstFlag==1):
                    self.mergedData=newData
                    firstFlag=0
                else:
                    self.mergedData = pd.merge(self.mergedData, newData, how='outer', on=['id']).to_sparse()
                
        return self.mergedData,self.categoricalIndex