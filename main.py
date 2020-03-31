import os
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import sklearn.datasets

"""
def bag_of_words(data):
	words = {}
	words_id = 0

	for i in range(0, len(data)):
		data[i] = data[i].lower()
		data[i] = re.sub(r'\W',' ', data[i])
		data[i] = re.sub(r'\s+',' ', data[i])

		tokens = nltk.word_tokenize(data[i])

		for token in tokens:
			if token not in words.keys():
				words[token] = words_id
				words_id += 1

	occ = [0] * len(data)

	for i in range(0, len(data)):
		tokens = nltk.word_tokenize(data[i])
		occ[i] = [0] * len(words)

		for token in tokens:
			occ[i][words[token]] += 1

	return words, occ
"""

"""
def word_frequencies(data, words, occ, tf_idf=False):
	t_occ = [0] * len(words)

	print(occ)
	if(tf_idf == True):
		print('Rien')
	else:
		for row in occ:
			for i in range(0, len(row)):
				t_occ[i] += row[i]

	freq = [[]] * len(occ)

	for i in range(0, len(freq)):
		freq[i] = [0.0] * len(occ[i])

		for j in range(0, len(freq[i])):
			freq[i][j] = (float)(occ[i][j] / t_occ[j])

	return freq
"""

def load_files():
	files = []
	cat = []
	categories = []
	catId = 0

	for file in os.listdir("20-newsgroups"):
		if file.endswith(".txt"):
			c = os.path.splitext(file)[0] # .split('.')[0]
			categories.append(c)
			files.append("20-newsgroups/" + file)
			cat.append(catId)
			catId += 1

	return { 'filenames': files, 'cat': cat }, categories

def read_file(filename, cat):
	with open(filename, encoding="utf8", errors="ignore") as f:
		reader = { 'data': f.read(), 'cat': cat }
	return reader

def bow(data, vocabulary=None):
	cv = CountVectorizer(vocabulary=vocabulary)
	occ = cv.fit_transform(data)
	return occ, cv.vocabulary_

def tf_idf(occ, use_idf=True):
	tf = TfidfTransformer(smooth_idf=True, use_idf=use_idf)
	return tf.fit_transform(occ)

def classifier(clf, train, test, files):
	clf.fit(train, files.target)
	return clf.predict(test)

def main():
	files = sklearn.datasets.load_files('20-newsgroups', encoding="utf8", decode_error='ignore');
	
	occ, vocabulary = bow(files.data)
	tfidf = tf_idf(occ)

	docs_new = [
		'The ball is remind me a sport', 
		'NASA will wait until 2040 to launch their fuse'
	]

	occ_test, _ = bow(docs_new, vocabulary)
	tfidf_test = tf_idf(occ_test)

	print(classifier(MultinomialNB(), tfidf, tfidf_test, files))

if __name__ == '__main__':
	main()