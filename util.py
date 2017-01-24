#coding=utf-8

import numpy as np
from collections import Counter

def count_sparsity(matrix,v = 0 ):
	'''
		count matrix sparsity.
	'''
	f = NoneElement()
	result = f(matrix).sum()
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



if __name__ == '__main__':
	m = np.zeros((2,2))
	print count_sparsity(m)
