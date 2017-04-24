# coding: utf-8
#
# data_preprocess.py
#
# Author: Huang Anbu
# Date: 2017.4
#
# Description: data processing, after execute this script, you will get new file "imdb.pkl", it
#   contains the following data structure:
#
#   - train_set: (train_set_x, train_set_y) tuple
#   - test_set: (test_set_x, test_set_y) tuple
#   - vocab: over 160000 extracted vocabulary words
#   - word2idx: word to index dictionary
#   - word2vec: word to vector dictionary
#
# Copyright©2017. All Rights Reserved. 
# ===============================================================================================

from basiclib import *

def get_dataset(path, stop_words):
	curdir = os.getcwd()
	
	train_set_pos_x, train_set_pos_y, train_set_neg_x, train_set_neg_y = [], [], [], []
	test_set_pos_x, test_set_pos_y, test_set_neg_x, test_set_neg_y = [], [], [], []
	sentences = []
	
	dir = os.path.abspath(os.path.join(path, "aclImdb", "train", "pos"))
	os.chdir(dir)
	for file in glob.iglob("*.txt"):
		with open(file, "rb") as fin:
			for line in fin:
				#sentences.append(line.strip().lower().decode('utf-8'))
				train_set_pos_x.append(line.strip().lower().decode('utf-8'))
	train_set_pos_x = [nltk.word_tokenize(sent) for sent in train_set_pos_x]
	train_set_pos_x = [[word for word in sent if word not in stop_words and len(word)>1] for sent in train_set_pos_x]
	train_set_pos_x = filter(lambda x: len(x)>10 and len(x)<1000, train_set_pos_x)
	train_set_pos_y = [1]*len(train_set_pos_x)
	print("positive train set size: %d"%(len(train_set_pos_x)))
	
	
	os.chdir(curdir)
	dir = os.path.abspath(os.path.join(path, "aclImdb", "train", "neg"))
	os.chdir(dir)
	for file in glob.iglob("*.txt"):
		with open(file, "rb") as fin:
			for line in fin:
				sentences.append(line.strip().lower().decode('utf-8'))
				train_set_neg_x.append(line.strip().lower().decode('utf-8'))
	train_set_neg_x = [nltk.word_tokenize(sent) for sent in train_set_neg_x]
	train_set_neg_x = [[word for word in sent if word not in stop_words and len(word)>1] for sent in train_set_neg_x]
	train_set_neg_x = filter(lambda x: len(x)>10 and len(x)<1000, train_set_neg_x)
	print("negative train set size: %d"%(len(train_set_neg_x)))
	train_set_neg_y = [0]*(len(train_set_neg_x))
	
	train_set_x = train_set_neg_x + train_set_pos_x
	train_set_y = train_set_neg_y + train_set_pos_y
	
	assert len(train_set_x)==len(train_set_y)
	print("train set size: %d"%len(train_set_x))
	
	
	
	os.chdir(curdir)
	dir = os.path.abspath(os.path.join(path, "aclImdb", "test", "pos"))
	os.chdir(dir)
	for file in glob.iglob("*.txt"):
		with open(file, "rb") as fin:
			for line in fin:
				sentences.append(line.strip().lower().decode('utf-8'))	
				test_set_pos_x.append(line.strip().lower().decode('utf-8'))
	test_set_pos_x = [nltk.word_tokenize(sent) for sent in test_set_pos_x]
	test_set_pos_x = [[word for word in sent if len(word)>1 and word not in stop_words] for sent in test_set_pos_x]
	test_set_pos_x = filter(lambda x: len(x)>10 and len(x)<1000, test_set_pos_x)
	test_set_pos_y = [1]*len(test_set_pos_x)
	print("positive test set size: %d"%(len(test_set_pos_x)))
	
	
	os.chdir(curdir)
	dir = os.path.abspath(os.path.join(path, "aclImdb", "test", "neg"))
	os.chdir(dir)
	for file in glob.iglob("*.txt"):
		with open(file, "rb") as fin:
			for line in fin:
				sentences.append(line.strip().lower().decode('utf-8'))	
				test_set_neg_x.append(line.strip().lower().decode('utf-8'))
	test_set_neg_x = [nltk.word_tokenize(sent) for sent in test_set_neg_x]
	test_set_neg_x = [[word for word in sent if len(word)>1 and word not in stop_words] for sent in test_set_neg_x]
	test_set_neg_x = filter(lambda x: len(x)>10 and len(x)<1000, test_set_neg_x)
	print("negative test set size: %d"%(len(test_set_neg_x)))
	test_set_neg_y = [0]*(len(test_set_neg_x))
	
	test_set_x = test_set_pos_x + test_set_neg_x
	test_set_y = test_set_pos_y + test_set_neg_y
	assert len(test_set_x)==len(test_set_y)
	print("test set size: %d"%(len(test_set_x)))
	
	sentences = train_set_x + test_set_x
	
	vocab = set(itertools.chain(*sentences))
	print("vocabulary size: %d" % len(vocab))
	os.chdir(curdir)
	
	return train_set_x, train_set_y, test_set_x, test_set_y, vocab

def build_word2vec(bin, vocab):
	word_vecs = {}
	with open(bin, "rb") as f:
		header = f.readline()
		vocab_size, layer1_size = map(int, header.split())
		binary_len = numpy.dtype('float32').itemsize * layer1_size
		for line in xrange(vocab_size):
			word = []
			while True:
				ch = f.read(1)
				if ch == ' ':
					word = ''.join(word)
					word = word.decode('utf-8')
					break
				if ch != '\n':
					word.append(ch)   
			if word in vocab:
			   word_vecs[word] = numpy.fromstring(f.read(binary_len), dtype='float32')  
			else:
				f.read(binary_len)
	return word_vecs
	

def build_dictionary(vocab, word2vec):
	for word in vocab:
		if word not in word2vec:
			word2vec[word] = numpy.random.uniform(-0.25,0.25,300).astype(theano.config.floatX) 
	
	w = numpy.zeros(shape=(len(vocab)+1, 300)).astype(theano.config.floatX) 
	word2idx = {}
	for idx, (word, rep) in enumerate(word2vec.iteritems()):
		w[idx+1] = rep 
		word2idx[word] = idx+1
	return w, word2idx
	
if __name__ == "__main__":
	stop_words = map(lambda x : x.strip().lower().decode('utf-8'), list(stopwords.words("english")))
	with open("stop_words.txt", "rb") as fin:
		stop_words = stop_words + [line.strip().lower().decode('utf-8') for line in fin]
	#print(stop_words)
	
	dir, base = os.path.split(__file__)
	print("start to load dataset...  %s" % time.strftime("%Y-%m-%d %X", time.localtime()))
	train_set_x, train_set_y, test_set_x, test_set_y, vocab = get_dataset(dir, stop_words)
	print("end to load dataset...  %s\n\n" % time.strftime("%Y-%m-%d %X", time.localtime()))
	
	
	print("start to get word2vec representation for vocab...  %s" % time.strftime("%Y-%m-%d %X", time.localtime()))
	word2vec = build_word2vec("GoogleNews-vectors-negative300.bin", vocab)
	print("total vocabulary size: %d, %d words have word representation"%(len(vocab), len(word2vec)))
	print("end to get word2vec representation for vocab...  %s\n\n" % time.strftime("%Y-%m-%d %X", time.localtime()))
	
	
	print("start to build dictionary...  %s" % time.strftime("%Y-%m-%d %X", time.localtime()))
	word2vec, word2idx = build_dictionary(vocab, word2vec)
	print(word2idx['movie'])
	word2vec = numpy.array(word2vec).astype(theano.config.floatX)
	print(word2vec.shape)
	#print(numpy.sum(w[word2idx['movie']]), numpy.mean(w[word2idx['movie']]), w[word2idx['movie']][15])
	#print(numpy.sum(word2vec['movie']), numpy.mean(word2vec['movie']), word2vec['movie'][15])
	print("end to build dictionary...  %s\n\n" % time.strftime("%Y-%m-%d %X", time.localtime()))
	
	train_set = (train_set_x, train_set_y)
	test_set = (test_set_x, test_set_y)
	dump_data = (train_set, test_set, vocab, word2idx, word2vec)
	with open("imdb.pkl", "wb") as fout:
		cPickle.dump(dump_data, fout)
	
	
	
	
	
	
	