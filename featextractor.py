# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 08:36:19 2018

@author: LGPinto
"""

import pandas as pd
import csv
import numpy as np
import math
import scipy
import os
from geopy.distance import geodesic
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import hstack
from sklearn import manifold, neighbors
from collections import Counter
import Levenshtein
import scipy.cluster.hierarchy as hac
import datetime as dt
import itertools
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import networkx as nx
import random
from deepwalk import graph
import pyarrow.parquet as pq

class FeatureExtractor():
    def __init__(self, path, sample):
        self.path = path
        self.sample = sample
    
    def _loadData(self, file, sample, graph=False):
        if(graph):
            data = pq.ParquetDataset(file).read().to_pandas()
            return data.reset_index(drop=True)
        if(file[-3:]!='csv'):
            data = pq.ParquetDataset(file).read().to_pandas()
            return data[data['id'].isin(sample)].reset_index(drop=True)
        else:
            #check file size
            slices = 1
            if os.stat(file).st_size/1000000>2000:
                slices = 2
            if os.stat(file).st_size/1000000>10000:
                slices = 3
    
            csvfile = open(file, 'rt', encoding='mbcs')
            print('Counting CSV Rows...')
            n = sum(1 for line in csvfile) - 1
            gap = int(n/slices)
            csvfile.close()
            print('Number of lines: '+str(n))
            
            csvfile = open(file, 'rt', encoding='mbcs')
            rowReader = csv.reader(csvfile, delimiter=',', quotechar='"')
            header = next(rowReader)
            csvfile.close()
            
            rows=pd.DataFrame(columns=header)
            
            currentRow=1
            for curSlice in range(0,slices):
                csvfile = open(file, 'rt', encoding='mbcs')
                currentRows=[]
                rowReader = csv.reader(csvfile, delimiter=',', quotechar='"')
                if(currentRow+gap<=n):
                    limit=currentRow+gap
                else:
                    limit=n
                for index,row in enumerate(itertools.islice(rowReader, currentRow, limit)):
                    currentRows.append(row)
                currentRows = pd.DataFrame(currentRows,columns=header)
                rows = rows.append(currentRows[currentRows['id'].isin(sample)])
                currentRow+=gap  
                csvfile.close()
                    
            return rows.reset_index(drop=True)
            
    
    def _assignCodes(self, values):
        code = 0
        codes = dict()
        for value in values:
            codes[value]=code
            code+=1
        return codes
    
    @property
    def features(self):
        raise NotImplementedError

class TargetFeatureExtractor(FeatureExtractor):
    def __init__(self, path, target, sample):
        self.targetPath=target
        super().__init__(path, sample)
    
    @property
    def features(self):
        targetPath = self.targetPath
        idPath = self.path
        
        print("Importing Id Dataset")
        users = self._loadData(idPath,self.sample)
        
        ids = users.id.unique()
        
        usersWithChurn = pd.DataFrame({'id': ids, 'churn': [0]*len(ids)})
        usersWithChurn = usersWithChurn.set_index('id')
        
        print("Importing Target Dataset")
        retainedUsers = self._loadData(targetPath,self.sample)
        
        retainedIds = retainedUsers.id.unique()
        
        print("Generating Id Target Matrix")
        for userId in ids:
           if userId not in retainedIds:
               usersWithChurn.at[userId] = 1
        
        idChurn = pd.DataFrame({'id': list([str(row) for row in ids]), 'churn': usersWithChurn['churn']})
        
        return idChurn.to_sparse()
    
class BehavFeatureExtractor(FeatureExtractor):
    def __init__(self, path, sample):
        super().__init__(path, sample)
        
class SocialGraphFeatureExtractor(BehavFeatureExtractor):
    def __init__(self, path, sample, timestamps,takenNodes=None):
        self.takenNodes=takenNodes
        self.timestamps=timestamps
        super().__init__(path, sample)
        
    def _encodeNode(self,node,dictionary):
        if(node is not None):
            if(node in dictionary.keys()):
                return dictionary[node]
            else:
                return len(dictionary.values())+1
        else:
            return 0
            
    
    @property
    def features(self, number_walks = 1, walk_length = 100, 
                        max_memory_data_size = 1000000000, seed = 0, 
                        representation_size = 64, window_size = 5,
                        workers = 1, output = "SocialGraph",
                        vertex_freq_degree = False, nodes=None, take=100):
        
        path = self.path
        takenNodes = self.takenNodes
        
        print("Importing Social Graph Dataset")
        edges = self._loadData(path,self.sample,graph=True)
        
        print("Generating List of Edges")
        edgeTuples = edges.itertuples(False)
        
        del(edges)
        
        print("Generating Graph")
        socialNet = nx.DiGraph()
        sanitizedEdgeTuples=[]
        for edge in edgeTuples:
            sanitizedEdgeTuples.append(tuple(reversed(edge)))
        
        socialNet.add_edges_from(sanitizedEdgeTuples)
        del(edgeTuples)
        
        print("Computing Centrality Measures")
        #k = int(socialNet.number_of_nodes()*0.001)
        
        inDegree = dict(socialNet.in_degree())
        outDegree = dict(socialNet.out_degree())
        
        maxInDegree = max(inDegree.values())
        maxOutDegree = max(outDegree.values())
        
        minInDegree = min(inDegree.values())
        minOutDegree = min(outDegree.values())
        
        #closeness = nx.closeness_centrality(socialNet)
        #betweenness = nx.betweenness_centrality(socialNet,k=k)
        #eigenvector = nx.eigenvector_centrality(socialNet,k=k)
        
        # remove self loops
        print("Removing Self Loops")
        socialNet.remove_edges_from(socialNet.selfloop_edges())
        
        # convert to deepwalk format
        print("Transforming Graph Data Format")
        dwSocialNet = graph.from_networkx(socialNet, False)
        del(socialNet)
        
        # deepwalk process
        G = dwSocialNet
        del(dwSocialNet)
       
        num_walks = len(G.nodes()) * number_walks
        
        data_size = num_walks * walk_length
        
        if data_size < max_memory_data_size:
            print("Generating Random Walks")
            walks = graph.build_deepwalk_corpus(G, num_paths=number_walks, 
                                                path_length=walk_length, alpha=0, 
                                                rand=random.Random(seed))
        print("Applying Sampling Procedure")
        if(nodes is None):
            nodes = [walk[0] for walk in walks]
        nodeWalks = pd.DataFrame({'node':nodes})
        nodeWalks = pd.concat([nodeWalks, pd.DataFrame.from_records(walks)], axis=1)
        nodeWalkSample = nodeWalks[nodeWalks.node.isin(self.sample)]
        del(G)
        
        if(takenNodes==None):
            allNodes = []
            for walk in walks:
                for node in walk:
                    allNodes.append(node)
            
            nodeFrequencies = Counter(allNodes)
            
            del(allNodes)
            
            orderedNodeFrequencies = nodeFrequencies.most_common()
            
            del(nodeFrequencies)
            
            takenNodes = [node[0] for node in orderedNodeFrequencies[1:take]]
            
            del(orderedNodeFrequencies)
        
        print("Building One-Hot Encoded Matrix")
        encodedWalks = scipy.sparse.dok_matrix((len(nodeWalkSample['node'])*len(self.timestamps),take))
        nodeCodes = self._assignCodes(takenNodes)
        
        dayNodes = []
        row=0
        for node in nodeWalkSample['node']:
            #encode first row
            node_walk = nodeWalkSample.set_index('node').loc[node].tolist()
            other_flag=0
            col=0
            for element in takenNodes:
                if(element in node_walk):
                    encodedWalks[row,col]=1
                else:
                    encodedWalks[row,col]=0
                    other_flag=1
                col+=1
            encodedWalks[row,col]=other_flag
            row+=1
            encodedRow = encodedWalks[row]
            dayNodes.append((node,self.timestamps[0]))
            #encode all other rows for user
            for time in self.timestamps[1:]:
                dayNodes.append((node,time))
                encodedWalks[row] = encodedRow
                row+=1
        del(nodeCodes)

        num_rows, num_cols = encodedWalks.shape
        
        columns=[]
        for walk in range(0,num_cols):
            columns.append("node"+str(walk))

        dayNodes = pd.SparseDataFrame({'id':[row[0] for row in dayNodes],'timestamp':[row[1] for row in dayNodes], 'indegree':[(inDegree[int(row[0])] - minInDegree)/(maxInDegree - minInDegree) for row in dayNodes], 'outdegree':[(inDegree[int(row[0])] - minOutDegree)/(maxOutDegree - minOutDegree) for row in dayNodes]})
        graphDayNodes = pd.concat([dayNodes, pd.SparseDataFrame(encodedWalks,columns=columns)],axis=1)
        
        return takenNodes,graphDayNodes.to_sparse()

class SearchQueryFeatureExtractor(BehavFeatureExtractor):
    def __init__(self, path, sample, searchQueryDictionary=None):
        self.searchQueryDictionary=searchQueryDictionary
        super().__init__(path, sample)
        
    def _nearestDictionaryEntry(self, query, dictionary):
        distances = dict()
        for entry in dictionary:
            distances[entry] = Levenshtein.distance(query, entry)
            if(distances[entry]==0):
                return entry
        return min(distances, key=distances.get)
    
    @property
    def features(self, take=200, uniqueTokens=None):
        path = self.path
        searchQueryDictionary=self.searchQueryDictionary
        
        print("Importing Search Dataset")
        readUserSearches = self._loadData(path,self.sample)

        userSearches = readUserSearches.drop("unix_timestamp", 1)
        
        aggregatedUserSearches = userSearches.groupby("id")["query"].apply(list)
        
        print("Counting Searches per User")
        searchCounts = userSearches.groupby(['id']).size().reset_index(name='searches')
        minSearches=min(searchCounts['searches'])
        maxSearches=max(searchCounts['searches'])
        normalizedSearches = ([(search - minSearches)/(maxSearches - minSearches) for search in searchCounts['searches']])
        del(searchCounts)
        del(minSearches)
        del(maxSearches)
        
        if(searchQueryDictionary==None):
            print("Generating Search Query Frequency Table")
            searchQueries = []
            for index, row in aggregatedUserSearches.to_frame().iterrows():
                userQueries = row._values[0]
                searchQueries.extend(userQueries)
            del(aggregatedUserSearches)   
            
                
            searchQueryFrequencies = Counter(searchQueries)
            del(searchQueries)
            
            orderedSearchQueryFrequencies = searchQueryFrequencies.most_common()
            
            takenQueries = [query[0] for query in orderedSearchQueryFrequencies[1:take]]
            
            takenQueryFrequencies = [query[1] for query in orderedSearchQueryFrequencies[1:take]]
            
            del(orderedSearchQueryFrequencies)
    
            print("Clustering Similar Search Queries")
            transformedQueries = np.array(takenQueries).reshape(-1,1)
            queryDistanceMatrix = pdist(transformedQueries, lambda a,b: Levenshtein.distance(a[0],b[0]))
            searchQueryLinkage = hac.linkage(queryDistanceMatrix, metric=lambda a,b: Levenshtein.distance(a[0],b[0]))
    
            print("Generating Search Spellchecking Dictionary")
            searchQueryDictionary = list(pd.DataFrame({'query': takenQueries, 'frequency': takenQueryFrequencies, 'cluster': hac.cut_tree(searchQueryLinkage, height=5)[:,0].tolist()}).groupby("cluster").apply(lambda x: x.sort_values("frequency", ascending=False).head(1))["query"])

        print("Applying Spellchecking Procedure")
        idTimestamps = pd.DataFrame({"id":userSearches["id"], "timestamp": readUserSearches['unix_timestamp'].apply(lambda x: dt.datetime.fromtimestamp(int(x)).strftime('%d%m%y'))})
        del(readUserSearches)
        
        normalizedSearches = pd.DataFrame({"id": [str(user) for user in userSearches.id.unique()],"searches":normalizedSearches})
        idTimestampedQueries = pd.DataFrame({"id":list([str(row) for row in userSearches["id"]]), "timestamp": idTimestamps["timestamp"],"queries":userSearches["query"].apply(lambda query: self._nearestDictionaryEntry(query,searchQueryDictionary)+' ')}).groupby(['id','timestamp']).agg({'id':'max','timestamp':'max','queries':'sum'})
        idTimestampedQueries = pd.merge(normalizedSearches,idTimestampedQueries, how='outer', on=['id'])
        del(userSearches)
        
        print("Building One-Hot Encoded Matrix")
        tokenizedQueries = pd.DataFrame({"id":idTimestampedQueries['id'],"timestamp":idTimestampedQueries['timestamp'],"queries":idTimestampedQueries.queries.apply(lambda query: list(filter(None, query.split(" "))))})
        
        if uniqueTokens is None:
            tokens = []
            for row in tokenizedQueries.queries:
                tokens.extend(row)
            uniqueTokens = list(set(tokens))
            del(tokens)
        
        tokenizedDictionary = [query.split(' ') for query in searchQueryDictionary]
        dictionaryTokens = []
        for query in tokenizedDictionary:
            dictionaryTokens.extend(query)
        dictionaryTokens=list(set(dictionaryTokens))    
            
        mlb = MultiLabelBinarizer(dictionaryTokens)
        oneHotEncodedQueries = mlb.fit_transform(tokenizedQueries.queries)
        
        queryCols = []
        col=0
        for queryCol in mlb.classes_:
            queryCols.append("query"+str(col))
            col+=1
        
        encodedTimestampedQueries = pd.concat([idTimestampedQueries[['id','timestamp','searches']].reset_index(drop=True), pd.DataFrame(oneHotEncodedQueries, columns=queryCols)],axis=1)
        #del(idTimestampedQueries)
        
        #return tokenizedQueries,encodedTimestampedQueries.to_sparse(),uniqueTokens
        return searchQueryDictionary,encodedTimestampedQueries.to_sparse()
    
class AppDownloadFeatureExtractor(BehavFeatureExtractor):
    def __init__(self, path, sample, takenPkg=None):
        self.takenPkg=takenPkg
        super().__init__(path, sample)

    def _binaryEncode(self, value):
        if value >0:
            return 1
        else:
            return 0
        
    def _applyCodes(self,dictionary,key):
        if(key in dictionary.keys()):
            return dictionary[key]
        else:
            return len(dictionary.values())
    
    @property
    def features(self, take=200):
        path = self.path
        takenPkg=self.takenPkg
        
        print("Importing App Download Dataset")
        sanitizedUsers = self._loadData(path,self.sample)
        
        print("Counting Downloads per User")
        downloadCounts = sanitizedUsers.groupby(['id']).size().reset_index(name='downloads')
        minDownloads=min(downloadCounts['downloads'])
        maxDownloads=max(downloadCounts['downloads'])
        normalizedDownloads = ([(download - minDownloads)/(maxDownloads - minDownloads) for download in downloadCounts['downloads']])
        del(downloadCounts)
        del(minDownloads)
        del(maxDownloads)
        
        if(takenPkg==None):
            print("Generating App Download Frequency Table")
            packageNames = sanitizedUsers.pkg
                  
            appDownloads = []
            row=0
            while row < len(packageNames):
                userDownloads = packageNames[row]
                appDownloads.append(userDownloads)
                row+=1
                
            del(packageNames)
            
            appDownloadFrequencies = Counter(appDownloads)
            
            del(appDownloads)
            
            orderedAppDownloadFrequencies = appDownloadFrequencies.most_common()
            
            del(appDownloadFrequencies)
            
            takenPkg = [app[0] for app in orderedAppDownloadFrequencies[1:take]]
        
            del(orderedAppDownloadFrequencies)
        
        print("Generating App Download Data Codes")
        pkgNameCodes = self._assignCodes(takenPkg)
        
        print("Encoding App Download Data")
        filteredIndexes = []
        filteredRows = sanitizedUsers[sanitizedUsers["pkg"].isin(takenPkg)]
        for index in filteredRows.index:
            filteredIndexes.append(index)
                
        codedDownloads = pd.DataFrame({"packageName": [self._applyCodes(pkgNameCodes,pkg) for pkg in filteredRows["pkg"]]})
        
        enc = OneHotEncoder(len(pkgNameCodes.keys())+1)
        oneHotEncodedDownloads = enc.fit_transform(codedDownloads)
        
        del(codedDownloads)
        
        print("Generating User Timestamp Hashs")
        userIdCodes = dict()
        counter=0
        for index in filteredIndexes:
            userIdCodes[counter]=(dt.datetime.fromtimestamp(int(str(sanitizedUsers["unix_timestamp"][index]))).strftime('%d%m%y')+str(sanitizedUsers["id"][index]))
            counter+=1
            
        print("Aggregating Data per User")
        pairedBehaviors = []
        index=0
        while index < len(userIdCodes):
            pairedBehaviors.append((userIdCodes[index],oneHotEncodedDownloads[index]))
            index+=1
            
        del(userIdCodes)
        
        uniqueIds = []
        aggregatedFeatures = []
        for key, group in itertools.groupby(sorted(pairedBehaviors, key = lambda i: i[0]), lambda i: i[0]):
            uniqueIds.append(key)
            features = list(group)
            #we are aggregating per user per day, where each value is the sum per column
            aggregatedFeatures.append(np.add.reduce([i[1] for i in features]))

            
        del(pairedBehaviors)
        
        appCols = []
        col = 0
        for takenPackage in takenPkg:
            appCols.append("app"+str(col))
            col+=1
        appCols.append("app"+str(col))
        
        print("Preparing Raw Features")
        aggregatedRows = []
        for aggregatedRow in aggregatedFeatures:
            aggregatedRows.append(aggregatedRow.todense().tolist()[0])
        #transforming np matrix into pd dataframe, where each value will be binary encoded
        transformedDownloads = pd.DataFrame.from_records(aggregatedRows, columns=appCols).applymap(self._binaryEncode)
        
        del(aggregatedRows)
        
        print("Building One Hot Encoded Matrix")
        normalizedDownloads = pd.DataFrame({"id": [str(user) for user in sanitizedUsers.id.unique()],"downloads": normalizedDownloads})
        transformedIdTimestamp = pd.DataFrame({"id": [str(userIdCode).replace(str(userIdCode)[0:6],"") for userIdCode in uniqueIds], "timestamp": [str(userHash)[0:6] for userHash in uniqueIds]})
        transformedIdTimestamp = pd.merge(normalizedDownloads, transformedIdTimestamp, how='inner', on=['id'])
        del(uniqueIds)
        
        normalizedDownloads = pd.concat([transformedIdTimestamp, transformedDownloads],axis=1)
        
        return takenPkg,normalizedDownloads.to_sparse()
    
class ClickstreamFeatureExtractor(BehavFeatureExtractor):
    def __init__(self, path, sample, urlCodes=None, cardTypeCodes=None, sourceCodes=None, packageNameCodes=None):
        self.urlCodes=urlCodes
        self.cardTypeCodes=cardTypeCodes
        self.sourceCodes=sourceCodes
        self.packageNameCodes=packageNameCodes
        super().__init__(path, sample)
        
    def _applyCodes(self,dictionary,key):
        if(key in dictionary.keys()):
            return dictionary[key]
        else:
            return len(dictionary.values())
    
    @property
    def features(self, take=50, ):
        path = self.path
        urlCodes=self.urlCodes
        cardTypeCodes=self.cardTypeCodes
        sourceCodes=self.sourceCodes
        packageNameCodes=self.packageNameCodes
        
        print("Importing Timeline Events Dataset")
        userBehaviors = self._loadData(path,self.sample)
        
        specificData = userBehaviors["specific"]

        #Assigning User Hash (ddmmyy+userId) to DF Index
        userIdCodes = dict()
        for index in userBehaviors.index:
            userIdCodes[index]=dt.datetime.fromtimestamp(int(str(userBehaviors["unix_timestamp"][index]))).strftime('%d%m%y')+str(userBehaviors["id"][index])
                      
        print("Sanitizing Rows")
        packageNames = []
        urls = []
        for row in specificData:
            packageNames.append(row["app"])
            urls.append(row["url"])
        
        del(specificData)
        
        print("Building Final Dataframe")
        sanitizedUserBehaviors = pd.DataFrame({"cardType": userBehaviors["card_type"], "source": userBehaviors["source"], "packageName": packageNames, "url": urls})

        if(urlCodes == None and cardTypeCodes == None and sourceCodes == None and packageNameCodes == None):
            print("Encoding Nominal Variables")
            uniqueCardTypes = list(set(sanitizedUserBehaviors["cardType"]))
            sources = sanitizedUserBehaviors["source"]
            del(sanitizedUserBehaviors)
            
            urlFrequencies = Counter(urls)
            packageNameFrequencies = Counter(packageNames)
            sourceFrequencies = Counter(sources)
            
            orderedUrls = urlFrequencies.most_common()
            orderedPackageNames = packageNameFrequencies.most_common()
            orderedSources = sourceFrequencies.most_common()
            del(urlFrequencies)
            del(packageNameFrequencies)
            del(sourceFrequencies)
            
            takenUrls = [url[0] for url in orderedUrls[1:take]]
            takenPackageNames = [packageName[0] for packageName in orderedPackageNames[1:take]]
            takenSources = [source[0] for source in orderedSources[1:take]]
            del(orderedUrls)
            del(orderedPackageNames)
            del(orderedSources)
            
            uniqueUrls = list(set(takenUrls))
            uniquePackageNames = list(set(takenPackageNames))
            uniqueSources = list(set(takenSources))
            del(takenUrls)
            del(takenPackageNames)
            del(takenSources)
            
            urlCodes = self._assignCodes(uniqueUrls)
            cardTypeCodes = self._assignCodes(uniqueCardTypes)
            sourceCodes = self._assignCodes(uniqueSources)
            packageNameCodes = self._assignCodes(uniquePackageNames)
        
            del(uniqueUrls)
            del(uniqueCardTypes)
            del(uniqueSources)
            del(uniquePackageNames)

        codedUserBehaviors = pd.DataFrame({"cardType": [self._applyCodes(cardTypeCodes,cardType) for cardType in userBehaviors["card_type"]], "source": [self._applyCodes(sourceCodes,source) for source in userBehaviors["source"]], "packageName": [self._applyCodes(packageNameCodes,packageName) for packageName in packageNames], "url": [self._applyCodes(urlCodes,url) for url in urls]})
        
        enc = OneHotEncoder(len(packageNameCodes.keys())+1)
        oneHotEncodedBehaviors = enc.fit_transform(codedUserBehaviors.drop('cardType',1))
        
        enc = OneHotEncoder(len(cardTypeCodes.keys())+1)
        oneHotEncodedCardType = enc.fit_transform(codedUserBehaviors['cardType'].reshape(-1, 1))

        oneHotEncodedBehaviors = hstack([oneHotEncodedBehaviors,oneHotEncodedCardType])
        num_rows, num_cols = oneHotEncodedBehaviors.shape
        
        oneHotEncodedBehaviors = oneHotEncodedBehaviors.asformat('csr')        
        
        del(codedUserBehaviors)
        
        print("Aggregating Data per User")
        pairedBehaviors = []
        for index in userBehaviors.index:
            pairedBehaviors.append((userIdCodes[index],oneHotEncodedBehaviors[index]))
            
        del(userBehaviors)
        
        #len(list(group)) is number of clicks
        uniqueIds = []
        clicks = []
        aggregatedFeatures = []
        for key, group in itertools.groupby(sorted(pairedBehaviors, key = lambda i: i[0]), lambda i: i[0]):
            features = list(group)
            uniqueIds.append(key)
            clicks.append(len(features))
            aggregatedFeatures.append(np.divide(np.add.reduce([i[1] for i in features]),len(features)))
        del(pairedBehaviors)
        
        
        
        columns=[]
        for column in range(0,num_cols):
            columns.append("click"+str(column))
        
        print("Preparing Raw Features")
        aggregatedRows = []
        for aggregatedRow in aggregatedFeatures:
            aggregatedRows.append(aggregatedRow.todense().tolist()[0])
        del(aggregatedFeatures)
        transformedFeatures = pd.DataFrame.from_records(aggregatedRows, columns=columns)
        del(aggregatedRows)
        
        #max min normalization of clicks
        minClicks=min(clicks)
        maxClicks=max(clicks)
        normalizedClicks = ([(click - minClicks)/(maxClicks - minClicks) for click in clicks])
        
        idClicks = pd.DataFrame({"id": [str(userIdCode).replace(str(userIdCode)[0:6],"") for userIdCode in uniqueIds], "timestamp": [str(userHash)[0:6] for userHash in uniqueIds], "clicks": normalizedClicks})
        normalizedUserBehaviors = pd.concat([idClicks, transformedFeatures],axis=1)

        return urlCodes, cardTypeCodes, sourceCodes, packageNameCodes, normalizedUserBehaviors.to_sparse()

class GeoFeatureExtractor(FeatureExtractor):
    def __init__(self, path, sample):
        super().__init__(path, sample)
    
    def _train_validate_test_split(self,df, train_percent=.6, validate_percent=.2, seed=None):
        np.random.seed(seed)
        perm = np.random.permutation(df.index)
        m = len(df.index)
        train_end = int(train_percent * m)
        validate_end = int(validate_percent * m) + train_end
        train = df.ix[perm[:train_end]]
        validate = df.ix[perm[train_end:validate_end]]
        test = df.ix[perm[validate_end:]]
        return train, validate, test
    
    def _isSanitized(self,a):
        try:
            float(a)
        except ValueError:
            return False
        if(a is ""):
            return False
        else:
            return True
        
    def _rSquared(self,y_,y):
        y_bar = np.mean(y)
        SST = sum(pd.Series(y).apply(lambda a: math.pow((a - y_bar),2)))
        SSReg = sum(pd.Series(y_).apply(lambda a: math.pow((a - y_bar),2)))
        return SSReg/SST
    
    @property
    def features(self, sampleRate = 0.05):
        path = self.path
        print("Importing Geographic Dataset")
        readGeoData = self._loadData(path,self.sample)
        
        userCoordinates = readGeoData[["id","latitude","longitude"]]
        del(readGeoData)
        
        print("Sanitizing Data")
        latitudeFilter = [self._isSanitized(lat) for lat in userCoordinates.latitude]
        longitudeFilter = [self._isSanitized(lon) for lon in userCoordinates.longitude]
        finalFilter = [lat and lon for lat, lon in zip(latitudeFilter, longitudeFilter)]
        sanitizedData = userCoordinates[finalFilter]
        
        print("Collecting Random Subsample")
        userCoordinateSample = sanitizedData.sample(int(len(sanitizedData)*sampleRate))
        userCoordinateSample = userCoordinateSample[["latitude","longitude"]]
        
        print("Generating Geographic Distance Matrix")
        transformedCoordinates = np.array(userCoordinateSample).astype(np.float)
        
        geoDistanceMatrix = pdist(transformedCoordinates, lambda a,b: geodesic((math.radians(a[0]),math.radians(a[1])),(math.radians(b[0]),math.radians(b[1]))).meters)
        del(transformedCoordinates)
        
        print("Executing Multidimensional Scaling Procedure")
        reshapedGeoDistMatrix = squareform(geoDistanceMatrix)
        del(geoDistanceMatrix)
        seed = np.random.RandomState(seed=3)
        
        mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                           dissimilarity="precomputed", n_jobs=1)
        fittedMds = mds.fit(reshapedGeoDistMatrix)
        del(reshapedGeoDistMatrix)
        self.stress = fittedMds.stress_
        pos = fittedMds.embedding_
        
        print("Initiating Embedding Estimation for Entire Database")
        
        dataset = pd.DataFrame({"latitude": userCoordinateSample["latitude"], "longitude": userCoordinateSample["longitude"], "Y1":[element[0] for element in pos.tolist()], "Y2":[element[1] for element in pos.tolist()]})
        
        print("Training Model")
        training, validation, test = self._train_validate_test_split(dataset, train_percent=0.70, validate_percent=0.15)
        
        trainX = training[["latitude","longitude"]]
        trainY1 = training["Y1"]
        trainY2 = training["Y2"]
        
        validationX = validation[["latitude","longitude"]]
        validationY1 = validation["Y1"]
        validationY2 = validation["Y2"]
        
        testX = test[["latitude","longitude"]]
        testY1 = test["Y1"]
        testY2 = test["Y2"]
        
        print("Validating Model")
        n_neighbors = 5
        knn11 = neighbors.KNeighborsRegressor(n_neighbors, weights="distance")
        knn12 = neighbors.KNeighborsRegressor(n_neighbors, weights="distance")
        
        n_neighbors = 7
        knn21 = neighbors.KNeighborsRegressor(n_neighbors, weights="distance")
        knn22 = neighbors.KNeighborsRegressor(n_neighbors, weights="distance")
        
        n_neighbors = 11
        knn31 = neighbors.KNeighborsRegressor(n_neighbors, weights="distance")
        knn32 = neighbors.KNeighborsRegressor(n_neighbors, weights="distance")
        
        #Validation
        validationPredictedY11 = knn11.fit(trainX, trainY1).predict(validationX)
        validationPredictedY12 = knn12.fit(trainX, trainY2).predict(validationX)
        validationPredictedY21 = knn21.fit(trainX, trainY1).predict(validationX)
        validationPredictedY22 = knn22.fit(trainX, trainY2).predict(validationX)
        validationPredictedY31 = knn31.fit(trainX, trainY1).predict(validationX)
        validationPredictedY32 = knn32.fit(trainX, trainY2).predict(validationX)
        
        rSquared11 = self._rSquared(validationPredictedY11,validationY1)
        rSquared12 = self._rSquared(validationPredictedY12,validationY2)
        rSquared1 = np.mean([rSquared11, rSquared12])
        
        rSquared21 = self._rSquared(validationPredictedY21,validationY1)
        rSquared22 = self._rSquared(validationPredictedY22,validationY2)
        rSquared2 = np.mean([rSquared21, rSquared22])
        
        rSquared31 = self._rSquared(validationPredictedY31,validationY1)
        rSquared32 = self._rSquared(validationPredictedY32,validationY1)
        rSquared3 = np.mean([rSquared31, rSquared32])
        
        if rSquared1 == max([rSquared1,rSquared2,rSquared3]):
            knn1 = knn11
            knn2 = knn12
            print("Best K=5")
            #print("Best R-squared: "+str(rSquared1))
        elif rSquared2 == max([rSquared1,rSquared2,rSquared3]):
            knn1 = knn21
            knn2 = knn22
            print("Best K=7")
            #print("Best R-squared: "+str(rSquared2))
        else:
            knn1 = knn31
            knn2 = knn32
            print("Best K=11")
            #print("Best R-squared: "+str(rSquared3))
        
        del(validationPredictedY11)
        del(validationPredictedY12)
        del(validationPredictedY21)
        del(validationPredictedY22)
        del(validationPredictedY31)
        del(validationPredictedY32)
        
        print("Testing Model")
        #Test
        testPredictedY1 = knn1.fit(trainX, trainY1).predict(testX)
        testPredictedY2 = knn2.fit(trainX, trainY2).predict(testX)
        finalRSquared1 = self._rSquared(testPredictedY1,testY1)
        finalRSquared2 = self._rSquared(testPredictedY2,testY2)
        finalRSquared = np.mean([finalRSquared1, finalRSquared2])
        print("Final R-Squared: "+str(finalRSquared))
        
        del(testPredictedY1)
        del(testPredictedY2)
        
        print("Deploying Model")
        #Deployment
        finalModel1 = knn1.fit(trainX, trainY1)
        finalModel2 = knn2.fit(trainX, trainY2)
        finalPos1 = finalModel1.predict(sanitizedData[['latitude','longitude']])
        finalPos2 = finalModel2.predict(sanitizedData[['latitude','longitude']])
        
        print("Normalizing Position Vectors")
        normalizedPos1 = (finalPos1-min(finalPos1))/(max(finalPos1)-min(finalPos1))
        normalizedPos2 = (finalPos2-min(finalPos2))/(max(finalPos2)-min(finalPos2))
        
        normalizedIdPos = pd.DataFrame({"id":list([str(row) for row in sanitizedData['id']]), "x": normalizedPos1, "y": normalizedPos2})
                
        return normalizedIdPos.to_sparse()
    
class TechnoFeatureExtractor(FeatureExtractor):
    def __init__(self, path, sample, deviceCodes=None, androidVersionCodes=None):
        self.deviceCodes=deviceCodes
        self.androidVersionCodes=androidVersionCodes
        super().__init__(path, sample)
    
    def _deviceEncoder(self, device, deviceList):
        if(device in deviceList.keys()):
            return deviceList[device]
        else:
            return deviceList["other"]
        
    def _versionEncoder(self, version, versionList):
        if(version in versionList.keys()):
            return versionList[version]
        else:
            return versionList["other_ver"]
    
    @property
    def features(self):
        path = self.path
        deviceCodes=self.deviceCodes
        androidVersionCodes=self.androidVersionCodes
        
        print("Importing Technographic Dataset")
        userDevices = self._loadData(path,self.sample)
        sanitizedUserDevices = userDevices
        
        #Devices
        devices = sanitizedUserDevices.device.unique()
        
        print("Generating User Device Frequency Table")
        userDevices = []
        row=0
        while row < len(devices):
            userDevice = devices[row]
            userDevices.append(userDevice)
            row+=1
        userDeviceFrequencies = Counter(userDevices)
        orderedUserDeviceFrequencies = userDeviceFrequencies.most_common()
        
        take = 50
        takenDevices = [app[0] for app in orderedUserDeviceFrequencies[1:take]]
        takenDevices.append("other")
        
        if(deviceCodes==None):
            print("Generating User Device Codes")
            deviceCodes = self._assignCodes(takenDevices)
        
        print("Encoding User Device Data")
        codedDeviceList = []
        for device in sanitizedUserDevices["device"]:
            codedDeviceList.append(self._deviceEncoder(device, deviceCodes))
        
        codedDevices = pd.DataFrame({"device": codedDeviceList})
        
        enc = OneHotEncoder(len(deviceCodes.keys()))
        oneHotEncodedDevices = enc.fit_transform(codedDevices)
        
        #Android version
        print("Loading Android Version Data")
        versions = sanitizedUserDevices.android_version.unique()
        
        if(androidVersionCodes==None):
            print("Generating Android Version Codes")
            androidVersionCodes = self._assignCodes(versions)
            androidVersionCodes['other_ver'] = len(androidVersionCodes.keys())
        
        print("Encoding Android Version Data")
        codedAndroidVersionList = []
        for version in sanitizedUserDevices["android_version"]:
            codedAndroidVersionList.append(self._versionEncoder(version,androidVersionCodes))
        
        codedVersions = pd.DataFrame({"android_version": codedAndroidVersionList})
        
        enc = OneHotEncoder(len(androidVersionCodes.keys()))
        oneHotEncodedVersions = enc.fit_transform(codedVersions)
        
        androidCols=[]
        col=0
        for androidCol in androidVersionCodes.keys():
            androidCols.append("android"+str(col))
            col+=1
        
        deviceCols=[]
        col=0
        for deviceCol in deviceCodes.keys():
            deviceCols.append("device"+str(col))
            col+=1
        
        print("Building One Hot Encoded Matrix")
        normalizedTechnographics = pd.concat([pd.DataFrame({'id':[str(row) for row in sanitizedUserDevices.id]}), pd.DataFrame.from_records([row.todense().tolist()[0] for row in oneHotEncodedDevices], columns=deviceCols), pd.DataFrame.from_records([row.todense().tolist()[0] for row in oneHotEncodedVersions], columns=androidCols)],axis=1)
        
        return deviceCodes,androidVersionCodes,normalizedTechnographics.to_sparse()
    
class DemoFeatureExtractor(FeatureExtractor):
    def __init__(self, path, sample, languageCodes=None):
        self.languageCodes=languageCodes
        super().__init__(path, sample)
    
    def _applyCodes(self,dictionary,key):
        if(key in dictionary.keys()):
            return dictionary[key]
        else:
            return len(dictionary.values())
    
    @property
    def features(self):        
        languageCodes=self.languageCodes
        
        print("Importing Demographic Dataset")
        userDemographics = self._loadData(self.path,self.sample)
        sanitizedUserDemographics = userDemographics
        
        languages = sanitizedUserDemographics.language.unique()
        
        if(languageCodes==None):
            languageCodes = self._assignCodes(languages)
        
        print("Encoding User Language Data")
        codedLanguages = pd.DataFrame({"language": [self._applyCodes(languageCodes,language) for language in sanitizedUserDemographics["language"]]})
        
        enc = OneHotEncoder(len(languageCodes.keys())+1)
        oneHotEncodedLanguages = enc.fit_transform(codedLanguages)
        
        demoCols=[]
        col=0
        for demoCol in languageCodes.keys():
            demoCols.append("demo"+str(col))
            col+=1
        demoCols.append("demo"+str(col))
        
        print("Building One Hot Encoded Matrix")
        normalizedLanguages = pd.concat([pd.DataFrame({'id':[str(row) for row in sanitizedUserDemographics['id']]}), pd.DataFrame.from_records([row.todense().tolist()[0] for row in oneHotEncodedLanguages],columns=demoCols)],axis=1)
        
        return languageCodes,normalizedLanguages.to_sparse()