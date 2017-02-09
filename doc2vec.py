#coding=utf-8

### try to combine the user query matrix

import matplotlib
matplotlib.use('Agg')  #,图形并没有在屏幕上显示,但是已保存到文件,
import numpy as np
from nltk import word_tokenize
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec





def getSen(content):
	return [ LabeledSentence(words=word_tokenize(line),tags=[str(row)])    for row,line in enumerate(content)]

def train(content,modelfile):
	sentences = []
	sentences += getSen(content)

	global model
	model = Doc2Vec(alpha=0.025,min_alpha=0.025)
	model.build_vocab(sentences)
	for epoch in range(10):
		model.train(sentences)
		model.alpha -= 0.002
		model.min_alpha = model.alpha
	model.save(modelfile)
	# print model.most_similar(u'government')

model = None

def getMatrix(matrixfile,modelfile):
	global model
	if model == None:
		model = Doc2Vec.load(modelfile)
	np.savetxt(matrixfile,model.docvecs.doctag_syn0)
	return model.docvecs.doctag_syn0

def getvector(string,modelfile):
	global model
	if model == None:
		model = Doc2Vec.load(modelfile)
	return model.infer_vector(word_tokenize(string))

def checkModel(modelfile):
	global model
	if model == None:
		model = Doc2Vec.load(modelfile)
	print model.most_similar(u'government')



if __name__ == '__main__':

	#checkModel('/home/yangying/data/enwiki_dbow/doc2vec.bin')
	print getvector(u'Fukushima children that diagnosed with thyroid cancer but radiation is said to be an ¡®unlikely?','/home/yangying/data/enwiki_dbow/doc2vec.bin')

