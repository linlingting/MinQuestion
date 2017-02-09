#coding=utf-8

### this file is to calculate some stat about user topic origin csv

import pandas as pd
from collections import Counter
import sys


def queryAnswer(df):
	querylist = df.columns.values
	querylist = list(querylist)[1:-1]
	print 'total',len(querylist)
	count = 0
        for col in querylist[:]:
                vote = list(df[col])
		vd = dict(Counter(vote))
		if len(vd) > 2:	
			count += 1
			#print vd['yes'],vd['no']
			#print vd[1],vd[0]
	print 'hit',count

if __name__ == '__main__':
	if len(sys.argv) == 1:
		print 'need a filename'
		sys.exit()
	filename = sys.argv[1]
        #df = pd.read_csv('./user_topic_origin.csv')
        df = pd.read_csv(filename)
        queryAnswer(df)

