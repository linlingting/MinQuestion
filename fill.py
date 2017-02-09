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
import doc2vec,util
import copy,random
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

USER_NUM = 16739

Path = '/home/linlt/code/new_crawl/fill/'

def group(label):
	username = []
	df = pd.read_csv(Path + '/fulldata_xmeans.csv')
	#df = pd.read_csv( './fulldata.csv')
	row,col = df.as_matrix().shape
	USER_NUM = row
	f=df.iloc[:,[0,15]]
	df = pd.read_csv(Path + '/topic_matric_origin.csv')
	#df = pd.read_csv('./user_topic_origin.csv')
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
	match = {}
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
				match[str(i)+','+str(j)]=cos[i][j]
				data.append([i,j])
				# print querylist[i]
				# print interset
				# print querylist[j]
				# print '\n'
			j += 1
		i += 1
	# print 'match:',len(match)
	return match,data

def analy_doc2vec(cos,threshold = 0.85):
	row,col= cos.shape
	# print row,col
	data = []
	match = {}
	i = 0
	while i < row:
		j = i + 1
		while j < col:
			if cos[i][j] > threshold:
				data.append([i,j])
				match[str(i)+','+str(j)]=cos[i][j]
			j += 1
		i += 1
	# print 'match:',len(match)
	return match,data

	

def context1(querylist,username):
	directory = Path + '/comments/'
	toadd = {}
	for filename in os.listdir(directory):
		filepath = directory + '/' + filename
		lines = [l.strip().decode('utf-8','ignore') for l in open(filepath).readlines()]
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

	
def fill(rule,matrix,method = 'rake',threshold = 0.6):
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

	df = pd.read_csv(Path + '/fulldata_xmeans.csv')
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
	if number == 0:
		print "There is no user has these attributes!!"
	else:
		cluster = mode(cluster).mode[0]
		print 'cluster:',cluster

		df = pd.read_csv(Path + '/topic_matric_origin.csv')
		querylist = df.columns.values[1:-1]  #第一行
		querylist = [q.strip().decode('utf-8','ignore')  for q in list(querylist)]
		

		username = group(cluster)  ##同为0组的用户
		content = [ c.strip().decode('utf-8','ignore') for c in context1(querylist,username)]
		corpus = []
		if method == 'rake':
			Rake = RAKE.Rake('./SmartStoplist.txt')
			for i,q in enumerate(content):
				keyword = Rake.run(q) ##返回关键字
				corpus.append(' '.join([t[0].replace('-',' ') for t in keyword])) #关键字合并
			print corpus
			countvect = CountVectorizer()
			m = countvect.fit_transform(corpus)
			m = m.toarray()
			sim = cosine_similarity(m,m) #计算相似性
			edge,data_1 = analy(sim,querylist,corpus,threshold = threshold)

		elif method == 'doc2vec':
			filemodel = '/home/linlt/code/cluster' + '/wiki_doc2vec.bin'
			# doc2vec.train(content,filemodel)
			#m = doc2vec.getMatrix('doc2vec.matrix',filemodel)
			vectors = [list(doc2vec.getvector(q,filemodel)) for q in content]
			m = np.asarray(vectors)
			sim = cosine_similarity(m,m) #计算相似性
			edge,data_1 = analy_doc2vec(sim,threshold = threshold)

		print sim.shape


		print "query+comment match:",len(data_1)

		

		print 'matrix sparsity:',util.count_sparsity(matrix.as_matrix())
		#count = fill_with_col_ap(matrix,querylist,edge,data_1,sim,value = 'binary')
		count = fill_with_col_ap(matrix,querylist,edge,data_1,sim,value = 'float')
		#count = fill_with_ap(matrix,querylist,edge,data_1,sim)
		#matrix = fill_with_col_merge(matrix,querylist,edge,data_1,sim)
		
		print 'matrix sparsity:',util.count_sparsity(matrix.as_matrix())
		return matrix

def generate_match(querylist,data_1):
	tmp_match = { e:[e] for e in xrange(len(querylist))}
	for e in data_1:
		s,e = e[0],e[1]
		tmp_match[s].append(e)
		tmp_match[e].append(s)
	match = [ tmp_match[key] for key in tmp_match if len(tmp_match[key]) > 1]
	return match


def fill_with_col_merge(m,querylist,edge,data_1,sim):
	'''
	fill the matrix with merging the query
	with convergence
	'''
	tmp = m.as_matrix()
	cols = list(m.columns.values)
	match = generate_match(querylist,data_1)
	flags =  [True] * len(cols)
	row,col = tmp.shape
	for pair in match[::-1]:
		pair = [ p+1 for p in pair]
		node = pair[0]
		simnodes = pair[1:]
		if len(simnodes) == 0 : continue
		j = 0
		while j < row:
			if not util.isNone(tmp[j][node]):
				j += 1
				continue
			answers = tmp[j][simnodes]
			ans = util.most_commom(answers)
			if ans == None:
				j += 1
				continue
			tmp[j][node] = ans
			j += 1
		for simnode in simnodes:
			flags[simnode] = False
	qindexs = [ key for key,value in enumerate(flags) if value == True]
	questions = [cols[key] for key in qindexs]
	removed = [q for q in cols if q not in questions]
	tmp_copy = copy.copy(tmp)[:,qindexs]
	df = pd.DataFrame(data=tmp_copy,columns=questions)
	return df


def fill_with_col_ap(m,querylist,edge,data_1,sim,value='binary'):
	'''
	fill the matrix with ap,convergence
	iterate with col
	faster.
	value options: 'binary'--> yes/no
		       'float' --> numerical value
	'''
	tmp = m.as_matrix()
	match = generate_match(querylist,data_1)
	row,col = tmp.shape
	count = 0
	iter = 100
	i = 0
	changed = float('inf')
	tmp_copy = copy.copy(tmp)
	tmp_change = 0
	vd = util.ValueDict(edge)
	while i < iter and changed != 0:
		random.shuffle(match)
		changed = 0
		for pair in match:
			node = pair[0]
			simnodes = pair[1:]
			j = 0
			while j < row:
				if not util.isNone(tmp_copy[j][node]): 
					j += 1
					continue
				answers = tmp[j][simnodes]
				ans = None
				if value == 'binary':
					ans = util.most_commom(answers)
				elif value == 'float':
					ans = vd.getFloat(node,simnodes,answers)
				else:
					print 'wrong value'
					sys.exit()
				if ans == None or tmp[j][node] == ans: 
					j += 1
					continue
				tmp[j][node] = ans
				changed += 1
				j += 1
		print 'iteration: %d / changed: %d'%(i,changed)
		if i == 0: tmp_change = changed
		i += 1
	return tmp_change
	

def fill_with_ap(m,querylist,edge,data_1,sim):
	'''
	fill matrix with ap and convergence
	Not iterate by col
	Not recommend
	'''
	tmp = m.as_matrix()
	match = [[] for i in xrange(len(querylist))]
	for i in data_1:
		match[i[0]].append(i[1])
		match[i[1]].append(i[0])
	#print match
	row,col = tmp.shape
	i = 0
	count = 0
	tmp_copy = copy.copy(tmp)
	while i < row:
		diff = 1
		iter = 0
		while iter<100 and diff>0:
			diff = 0
			for node in xrange(len(querylist)):
				if (len(match[node]) > 0) and (util.isNone(tmp_copy[i][node+1])) : #tmp
					simset = match[node]
					# for node in simset:
						# 	if isNone(tmp[i][node+1]):
								## is nan,need to fill
					max_value = 0
					pos = -1
					for sim_node in simset:
						if sim[node][sim_node] > max_value and not util.isNone(tmp[i][sim_node+1]):
							pos = sim_node
							max_value = sim[node][sim_node]
					if pos == -1: ## all the similarity set is nan
							#print 'all the similar query is null'
						continue
					else:
						if tmp[i][node+1] != tmp[i][pos+1]:  ##最优解变了
							diff += 1
							if tmp[i][node+1]=='?':
								count += 1
							tmp[i][node+1] = tmp[i][pos+1]
							
				else: 	# no need to fill,already have value
					#print 'no need to fill'
					continue
				# if iter%1==0:
				# 	print"---> Iteration %d, changed: %f"%(iter,diff)
			iter += 1
		i += 1
	return count		



if __name__ == '__main__':
	#rule = {'Gender':['Male'],'Age':20,'Party':['Democratic Party'],'Occupation':['Student']}
	rule = {'Ethnicity':['White']}
	df = pd.read_csv(Path+'/topic_matric_origin.csv')
	method = 'rake'
	#method = 'doc2vec'
	new_df = fill(rule,df,method)
	new_df.to_csv('./test.csv',index=False)

