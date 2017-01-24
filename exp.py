#coding=utf-8

### try to combine the user query matrix

import matplotlib
matplotlib.use('Agg')  #,图形并没有在屏幕上显示,但是已保存到文件,
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import graphviz
import numpy as np
import RAKE
import json
import os
import csv
from scipy.stats import mode

USER_NUM = 1479
ATTRIBUTE = 16  ##14个属性加名字和
Path = '/home/linlt/code/cluster'

def group(label):
	username = []
	df = pd.read_csv(Path + '/fulldata.csv')
	f=df.iloc[:,[0,15]]
	df = pd.read_csv(Path + '/user_topic_origin.csv')
	f_1 = df.iloc[:,0]
	user = []
	for k in xrange(USER_NUM):
		user.append(f_1.iloc[k])
	class_label = label
	for k in xrange(USER_NUM):
		if f.iloc[k,1] == class_label:
			if f.iloc[k,0] in user:
				username.append(f.iloc[k,0])
	# print len(username)
	return username

def analy(cos,querylist,keywords,threshold = 0.6):
	row,col= cos.shape
	# print row,col
	match = []
	data = []
	i = 0
	while i < row:
		j = i + 1
		while j < col:
			if cos[i][j] > threshold:
				# print corpus[i],keywords[i]
				# print corpus[j],keywords[j]
				key = set(keywords[i].split()) & set(keywords[j].split())  ##关键字取交集
				# print key,cos[i][j]
				interset = ' '.join(key)
				match.append((i,j,interset))  ##相似度大的连接
				data.append((i,j))
				# print querylist[i]
				# print interset
				# print querylist[j]
				# print '\n'
			j += 1
		i += 1
	# print 'match:',len(match)
	return match,data

def draw(match,corpus):
	corpus = list(corpus)
	print len(match)
	dot = graphviz.Graph(format='pdf',engine='circo')
	i = 0
	for n in corpus[:]:
		dot.node(str(i),n)
		i += 1
	for m in match[:]:
		#dot.edge(corpus[m[0]],corpus[m[1]],m[2])
		dot.edge(str(m[0]),str(m[1]),m[2])
	output = file('xxx','w')
	tmp = [m[2] for m in match]
	output.write(' '.join(tmp))
	output.close()
	dot.render('query_small')
	

def context1(querylist,username):
	directory = Path + '/topic_comment'
	toadd = {}
	for filename in os.listdir(directory):
		filepath = directory + '/' + filename
		lines = [l.strip() for l in open(filepath).readlines()]
		for l in lines:
			l = l.replace('\\','/')
			dic = json.loads(l,strict= False)
			title = dic[u'title'].strip().encode('utf-8')
			text = dic[u'text'].strip().encode('utf-8')
			user = dic[u'name'].strip().encode('utf-8')
			if title in querylist and user in username:
				if title in toadd:
					tmp = toadd[title]
					tmp += text
					toadd[title] = tmp
				else:
					toadd[title] = text
	newquery = []
	for q in querylist:
		if q in toadd:
			newquery.append(q + ' ' + toadd[q])
		else:
			newquery.append(q)
	return newquery

	
def getdiff(rule,matrix):
	'''
	to call this function
	rule: dict-like object
	matrix: your specific matrix
	return a filled matrix
	'''
	attrs = []
	value = []
	for key, item in rule.items():
		attrs.append(key)
		if type(item) == list:
			value.append(item)
		else:
			value.append([item])

	df = pd.read_csv(Path + '/fulldata.csv')
	attribute = df.columns.values[:]  #第一行
	attribute = list(attribute)
	index = []
	for i in xrange(len(attrs)):
		index.append(attribute.index(attrs[i]))  ##cluster_user.csv多了用户名属性
	index.append(len(attribute)-1)  ##簇


	f=df.iloc[:,index]

	cluster = []
	number = 0
	for i in xrange(USER_NUM):
		num = 0
		for j in xrange(len(index)-1):
			if f.iloc[i,j] in value[j]:
				num += 1
		if num == len(index)-1:
			cluster.append(f.iloc[i,len(index)-1])
			number += 1
	print 'number:',number
	cluster = mode(cluster).mode[0]
	print 'cluster:',cluster

	df = pd.read_csv(Path + '/user_topic_origin.csv')
	querylist = df.columns.values[1:-1]  #第一行
	querylist = list(querylist)
	Rake = RAKE.Rake('./SmartStoplist.txt')
	# corpus = []
	# for i,q in enumerate(querylist):
	# 	keyword = Rake.run(q) ##返回关键字
	# 	corpus.append(' '.join([t[0].replace('-',' ') for t in keyword])) #关键字合并
	# countvect = CountVectorizer()
	# m = countvect.fit_transform(corpus)
	# m = m.toarray()
	# sim = cosine_similarity(m,m) #计算相似性
	# edge_2,data_2 = analy(sim,querylist,corpus)


	username = group(cluster)  ##同为0组的用户
	content = context1(querylist,username)
	print len(querylist)
	print len(content)
	corpus = []
	Rake = RAKE.Rake('./SmartStoplist.txt')
	for i,q in enumerate(content):
		keyword = Rake.run(q) ##返回关键字
		corpus.append(' '.join([t[0].replace('-',' ') for t in keyword])) #关键字合并
	countvect = CountVectorizer()
	m = countvect.fit_transform(corpus)
	m = m.toarray()
	sim = cosine_similarity(m,m) #计算相似性
	edge,data_1 = analy(sim,querylist,corpus)
	print sim.shape
	print len(corpus)

	corpus1 = []
	for i,q in enumerate(querylist):
		keyword = Rake.run(q) ##返回关键字
		corpus1.append(' '.join([t[0].replace('-',' ') for t in keyword])) #关键字合并
		
	countvect1 = CountVectorizer()
	m1 = countvect1.fit_transform(corpus1)
	m1 =m1.toarray()
	sim1 = cosine_similarity(m1,m1) #计算相似性
	edge1,data_11 = analy(sim1,querylist,corpus1)
	print len(data_1)
	print len(data_11)
	data_x = set(data_1)
	data_y = set(data_11)
	dis = data_x.difference(data_y)
	new = data_y.difference(data_x)
	for i in dis:
		print content[i[0]]
		print content[i[1]]
		print
	print '=========================='
	ti = 0
	for i in new:
		print content[i[0]]
		print content[i[1]]
		print corpus1[i[0]]
		print corpus1[i[1]]
		print
		ti += 1
		if ti > 50:break 


	
			


def isNone(value):
	if value == '?' or value == '0':
		return True
	return False

if __name__ == '__main__':
	#rule = {'Gender':['Male'],'Age':20,'Party':['Republican Party','Democratic Party']}
	rule = {'Gender':['Male'],'Age':20,'Party':['Democratic Party']}
	df = pd.read_csv('./user_topic_origin.csv')
	print getdiff(rule,df.as_matrix())


