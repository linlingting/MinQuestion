#coding=utf-8

import numpy as np

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
	

def NoneElement():
	return np.vectorize(isNone)



if __name__ == '__main__':
	m = np.zeros((2,2))
	print count_sparsity(m)
