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

def group(label):
	username = []
	df = pd.read_csv('./fulldata.csv')
	f=df.iloc[:,[0,15]]
	df = pd.read_csv('./user_topic_origin.csv')
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
		j = i
		while j < col:
			if i == j: 
				j+= 1
				continue
			if cos[i][j] > threshold:
				# print corpus[i],keywords[i]
				# print corpus[j],keywords[j]
				key = set(keywords[i].split()) & set(keywords[j].split())  ##关键字取交集
				# print key,cos[i][j]
				interset = ' '.join(key)
				if interset == '': 
					print keywords[i]
					print keywords[j]
					print

				match.append((i,j,interset))  ##相似度大的连接
				data.append([i,j])
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
	
def context(querylist,username):  ## 把评论加到query
	s = os.sep
	root = "./topic_comment" + s
	content = querylist[:]

	commentnum = 0
	for i in os.listdir(root):
		if os.path.isfile(os.path.join(root,i)):
			f = file(os.path.join(root,i))
			temp = ''
			line_num = 0
			title = 0
			for line in f.readlines():
				line_num += 1
				line = line.strip().replace('\\','/').replace('	','')
				dic = json.loads(line, strict=False)
				if line_num == 1 :
					if dic["title"] in querylist:
						if dic["name"].strip() in username:
							commentnum+=1
							title = 1
							temp += dic["title"]
							temp += ' '
							temp += dic["text"]
							temp += ' '
						else:
							continue
					else:
						break
				else:
					if dic["name"].strip() in username:
						if title == 1:
							temp += dic["text"]
							temp += ' '
						else:
							temp += dic["title"]
							temp += ' '
							temp += dic["text"]
							temp += ' '
							title = 1
					else:
						continue

			if len(temp) > 0 :  ##title 在 query 中出现
				ind = querylist.index(dic["title"])
				content[ind] = temp
			f.close()

	print "commentnum",commentnum  ##加入的评论数
	return content


if __name__ == '__main__':
	rule = raw_input("rule:")
	rule = eval(rule)
	attrs = []
	value = []
	for key, item in rule.items():
		attrs.append(key)
		value.append(item)

	df = pd.read_csv('./fulldata.csv')
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

	df = pd.read_csv('./user_topic_origin.csv')
	querylist = df.columns.values[1:-1]  #第一行
	querylist = list(querylist)
	# Rake = RAKE.Rake('./SmartStoplist.txt')
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
	content = context(querylist,username)
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



	print "query+comment match:",len(data_1)

	# num = 0
	# for i in data_1:
	# 	if i not in data_2:
	# 		print querylist[i[0]]
	# 		print edge[num][2]
	# 		print querylist[i[1]]
	# 		print '\n'
	# 	num += 1
	##匹配集合

	match = []

	while(len(data_1)>0):
		delete = []
		temp = set(data_1[0])
		for j in xrange(1,len(data_1)):
			if (data_1[j][0] in temp) or (data_1[j][1] in temp) :
				temp = temp | set(data_1[j])
				delete.append(data_1[j])
		data_1.remove(data_1[0])
		for i in delete:
			data_1.remove(i)
		if temp not in match:
			match.append(temp)
	# print match

	temp = ['user_topic']
	temp.extend(querylist)
	content = []
	content.append(temp)
	number_befor = 0
	number_after = 0
	line_num = 0
	with open('./user_topic_origin.csv') as f_3:
		for line in f_3:
			line_num += 1
			if line_num != 1:
				line = line.strip('\n')
				line = line.split(',')
				temp = ['?' for j in xrange(len(line))]
				for i in xrange(len(line)):
					if (line[i]=='yes') or (line[i]=='no'):
						number_befor += 1
					if line[i]!='?':
						temp[i] = line[i]
				for i in match: ##查看所有集合
					value = []  ##存放每个集合中有回答的query编号
					i = list(i)
					for j in i:
						if (line[j+1]=='yes') or (line[j+1]=='no'):
							value.append(j)
					if len(value) == 0: #一个集合里都没有值
						continue
					else:
						for j in i: #一个集合里存在的值
							if j not in value:
								max = 0.
								for k in value: ##找跟j相似度最大的点
									x = k
									if sim[j,k] > max:
										max = sim[j,k]
										x = k
									# print j,x,line[x+1],querylist

								temp[j+1] = line[x+1]
								number_after += 1
				content.append(temp)
	f_3.close()
	print "befor:", number_befor
	csvfile = file('./user_topic_merge.csv','wb')
	writer = csv.writer(csvfile)

	for i in xrange(len(content)):
		writer.writerow(content[i])

	number_after = 0
	line_num = 0
	with open('./user_topic_merge.csv') as f_3:
		for line in f_3:
			line_num += 1
			if line_num != 1:
				line = line.strip('\n')
				line = line.split(',')
				for i in xrange(len(line)):
					if (line[i]=='yes') or (line[i]=='no'):
						number_after += 1
	print "after:" ,number_after
