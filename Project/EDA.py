#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 20:33:39 2021

@author: wangziwen
"""

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

"""
read data
"""
dfm = pd.read_csv('/Users/wangziwen/Documents/Graduation/1st/Statistic Learning/project/marketdata_sample.csv')
dfn = pd.read_csv('/Users/wangziwen/Documents/Graduation/1st/Statistic Learning/project/news_sample.csv')

"""
clean market data
"""
dfm.info() #1 datetime, 2 object, 13 float 
#returnsClosePrevMktres1,returnsOpenPrevMktres1,returnsClosePrevMktres10,returnsOpenPrevMktres10
dfm.iloc[1,:]

#time
#2007~2016
dfm.time.describe() #object;only on 2007-02-01 22:00
dfm['time'] = dfm["time"].dt.strftime('%Y-%m-%d %H:%M')
dfm['time'] = pd.to_datetime(dfm.time, format='%Y-%m-%d %H')
dfm.set_index('time',inplace = True)

#assetCode & assetName
#note that a single company may have multiple assetCodes
dfm.assetCode.drop_duplicates() #no duplicate
dfm.assetName.drop_duplicates() #duplicat
dfm.assetName.value_counts().sort_values(ascending = False)
dfm[dfm.assetName=='Unknown'] # "Unknown" if the corresponding assetCode does not have any rows in the news data

#universe
dfm.universe.value_counts() #actually category
dfm['universe'] = dfm.universe.astype('category')
sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(figsize=(10,20))sns.countplot(x="universe", data=dfm)

#volume 右偏
dfm.volume.describe().astype('int')
sns.set_theme(style="darkgrid")
f, ax = plt.subplots(figsize=(10,20))
ax.ticklabel_format(style='plain', axis='x')
sns.displot(data=dfm, x="volume", kde=True)
plt.show()

#close 正態微右偏
dfm.close.describe().astype('int')
sns.set_theme()
fig, ax = plt.subplots(figsize=(10,20))
sns.displot(data=dfm, x="close", kde=True)

#open 正態微右偏 與close相似
dfm.open.describe().astype('int')
sns.set_theme()
#f, ax = plt.subplots(figsize=(10,20))
sns.displot(data=dfm, x="close", kde=True)
plt.show()

#returnsClosePrevRaw1
dfm.returnsClosePrevRaw1.describe()
sns.set_theme()
#f, ax = plt.subplots(figsize=(10,20))
sns.displot(data=dfm, x="returnsClosePrevRaw1", kde=True)
plt.show()

#returnsOpenPrevRaw1
dfm.returnsOpenPrevRaw1.describe()
#f, ax = plt.subplots(figsize=(10,20))
sns.displot(data=dfm, x="returnsOpenPrevRaw1", kde=True)
plt.show()

#returnsClosePrevMktres1 null
#This means that there may be instruments that enter and leave this subset of data. 
#There may therefore be gaps in the data provided, and this does not necessarily 
#imply that that data does not exist (those rows are likely not included due to the selection criteria)

#returnsOpenPrevMktres1 null

#returnsClosePrevRaw10
dfm.returnsClosePrevRaw10.describe()
sns.set_theme()
#f, ax = plt.subplots(figsize=(10,20))
sns.displot(data=dfm, x="returnsClosePrevRaw10", kde=True)
plt.show()

#returnsOpenPrevRaw10
dfm.returnsOpenPrevRaw10.describe()
sns.set_theme()
#f, ax = plt.subplots(figsize=(10,20))
sns.displot(data=dfm, x="returnsOpenPrevRaw10", kde=True)
plt.show()

#returnsClosePrevMktres10 null

#returnsOpenPrevMktres10 null

#returnsOpenNextMktres10
dfm.returnsOpenPrevRaw10.describe()
sns.set_theme()
#f, ax = plt.subplots(figsize=(10,20))
sns.displot(data=dfm, x="returnsOpenPrevRaw10", kde=True)
plt.show()

#corr open-close returnsOpenPrevRaw10-returnsClosePrevRaw10
corr = dfm.dropna(axis='columns').select_dtypes(include=['float64']).corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(7, 5))
    ax = sns.heatmap(corr,square=True, annot=True, cmap=cmap,vmin=-1,vmax=1, annot_kws={"size": 8},mask=mask)

"""
clean news data
"""
dfn.info() 
dfn.iloc[1,:]

#time
dfn['time'] = pd.to_datetime(dfn.time, format='%Y-%m-%d %H:%M')
dfn['time'] = dfn["time"].dt.strftime('%Y-%m-%d %H:%M:%S')
dfn.time.drop_duplicates()

#sourceTimestamp
dfn['sourceTimestamp'] = pd.to_datetime(dfn.time, format='%Y-%m-%d %H:%M')
dfn['sourceTimestamp'] = dfn["sourceTimestamp"].dt.strftime('%Y-%m-%d %H:%M:%S')
dfn.sourceTimestamp.drop_duplicates()

#firstCreated
dfn['firstCreated'] = pd.to_datetime(dfn.time, format='%Y-%m-%d %H:%M')
dfn['firstCreated'] = dfn["firstCreated"].dt.strftime('%Y-%m-%d %H:%M:%S')
dfn.firstCreated.drop_duplicates()

len(dfn[(dfn.time==dfn.sourceTimestamp)&(dfn.time==dfn.firstCreated)&(dfn.sourceTimestamp==dfn.firstCreated)])

#sourceId
dfn.sourceId.drop_duplicates()
dfn.sourceId.value_counts().sort_values(ascending=False)
dfn[dfn.sourceId=='23768af19dc69992'].iloc[:,16:]

#headline
dfn.headline.value_counts().sort_values(ascending=False)
dfn[dfn.headline=='Korea Hot Stocks-LG Elec, LG.Philips, Banks, KEPCO'].sourceId

#urgency
dfn.urgency.value_counts().sort_values(ascending=False)
dfn['urgency'] = dfn.urgency.astype('category')
sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(figsize=(10,20))
sns.countplot(x="urgency", data=dfn)

#takeSequence
dfn.takeSequence.value_counts().sort_values(ascending=False)
dfn['takeSequence'] = dfn.takeSequence.astype('category')
sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(figsize=(10,20))
sns.countplot(x="takeSequence", data=dfn)

#provider
dfn.provider.value_counts().sort_values(ascending=False)
dfn['provider'] = dfn.provider.astype('category')
sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(figsize=(10,20))
sns.countplot(x="provider", data=dfn)

#subjects
a = dfn.subjects[0]

#audiences

#bodySize
sns.set_theme()
sns.displot(data=dfn, x="bodySize", kde=True,height=5,aspect=5)
plt.show()

#companyCount 
sns.displot(data=dfn, x="companyCount", kde=True,height=5,aspect=5)
plt.show()

#headlineTag 
dfn.headlineTag.value_counts().sort_values(ascending=False)
dfn['headlineTag'] = dfn.headlineTag.astype('category')
sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(figsize=(10,20))
sns.countplot(x="headlineTag", data=dfn)

#marketCommentary  
dfn.marketCommentary.value_counts().sort_values(ascending=False)
dfn['marketCommentary'] = dfn.marketCommentary.astype('category')
sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(figsize=(10,20))
sns.countplot(x="marketCommentary", data=dfn)

#sentenceCount
sns.set_theme()
sns.displot(data=dfn, x="sentenceCount", kde=True,height=5,aspect=5)
plt.show()

#wordCount   
sns.set_theme()
sns.displot(data=dfn, x="wordCount", kde=True,height=5,aspect=5)
plt.show()

#assetCodes            100 non-null    object 
dfn.assetCodes.drop_duplicates() 
#assetName             100 non-null    object
a = dfn[['assetName','assetCodes']]
b = dfm[['assetName','assetCode']]
c = b.merge(a, on='assetName',how='left')

#firstMentionSentence 
sns.set_theme()
sns.displot(data=dfn, x="firstMentionSentence", kde=True,height=5,aspect=5)
plt.show()

#relevance  
sns.set_theme()
sns.displot(data=dfn, x="relevance", kde=True,height=5,aspect=5)
plt.show()

#sentimentClass 
dfn.sentimentClass.value_counts().sort_values(ascending=False)
dfn['sentimentClass'] = dfn.sentimentClass.astype('category')
sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(figsize=(10,20))
sns.countplot(x="sentimentClass", data=dfn)

dfn[dfn.sentimentClass==-1][['sentimentNegative','sentimentNeutral','sentimentPositive']]
#sentimentNegative  
sns.set_theme()
sns.displot(data=dfn, x="sentimentNegative", kde=True,height=5,aspect=5)
plt.show()

#sentimentNeutral 
sns.set_theme()
sns.displot(data=dfn, x="sentimentNeutral",height=5,aspect=5,hue='sentimentClass', multiple="stack")
plt.show()
 
#sentimentPositive   
sns.set_theme()
sns.displot(data=dfn, x="sentimentPositive", kde=True,height=5,aspect=5)
plt.show()


#sentimentWordCount
sns.set_theme()
sns.displot(data=dfn, x="sentimentWordCount", kde=True,height=5,aspect=5)
plt.show()

#noveltyCount12H, noveltyCount24H, noveltyCount3D, noveltyCount5D, noveltyCount7D
dfn.noveltyCount12H.value_counts().sort_values(ascending=False)
dfn.noveltyCount24H.value_counts().sort_values(ascending=False)
dfn.noveltyCount3D.value_counts().sort_values(ascending=False)
dfn.noveltyCount5D.value_counts().sort_values(ascending=False)
dfn.noveltyCount7D.value_counts().sort_values(ascending=False)
dfn['noveltyCount12H'] = dfn.noveltyCount12H.astype('category')
dfn['noveltyCount24H'] = dfn.noveltyCount24H.astype('category')
dfn['noveltyCount3D'] = dfn.noveltyCount3D.astype('category')
dfn['noveltyCount5D'] = dfn.noveltyCount5D.astype('category')
dfn['noveltyCount7D'] = dfn.noveltyCount7D.astype('category')

#volumeCounts12H , volumeCounts24H, volumeCounts3D, volumeCounts5D, volumeCounts7D
dfn.volumeCounts12H.value_counts().sort_values(ascending=False)
dfn.volumeCounts24H.value_counts().sort_values(ascending=False)
dfn.volumeCounts3D.value_counts().sort_values(ascending=False)
dfn.volumeCounts5D.value_counts().sort_values(ascending=False)
dfn.volumeCounts7D.value_counts().sort_values(ascending=False)

sns.set_theme()
sns.displot(data=dfn, x="volumeCounts24H", kde=True,height=5,aspect=5)
plt.show()
