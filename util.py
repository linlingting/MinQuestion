#coding=utf-8

import numpy as np
from collections import Counter

def count_sparsity(matrix,v = 0 ):
	'''
		count matrix sparsity.
	'''
	f = NoneElement()
	result = f(matrix).sum()
	if v == 1:
		print matrix.shape
		print result
	return 1 - result / float(np.prod(matrix.shape))

def isNone(value):
	if value == '?' or value == '0' or value == 0:
		return True
	return False

def most_commom(elements):
	d = Counter(elements)
	ele = d.most_common(1)[0][0]
	if isNone(ele):
		return None
	else:	
		return ele
	
	

def NoneElement():
	return np.vectorize(isNone)


def sysmbol(vote):
	if vote == 'yes':
		return 1
	elif vote == 'no':
		return -1
	elif vote == '?':
		return 0
	elif vote > 0:
		return 1
	else:
		return 0


def getKey(node1,node2):
	if node1 < node2:
		return str(node1) + ',' + str(node2)
	else:
		return str(node2) + ',' + str(node1)

class ValueDict:
	
	def __init__(self,match):
		self.match = match
	
	def getFloat(self,node,indexs,values):
		values = list(values)
		ans = 0
		for key,index in enumerate(indexs):
			polar = self.match[getKey(node,index)]
			ans += sysmbol(values[key]) * polar
		if ans != 0:
			return ans
		else:
			return None


if __name__ == '__main__':
	m = np.zeros((2,2))
	print count_sparsity(m)
