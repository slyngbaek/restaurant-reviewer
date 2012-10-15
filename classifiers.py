import nltk

# unigram classifier
def unigramFeatures(word):
   return {'unigram':word}

def unigramFeatureSets(data): #data accepted as (rating, list of words)
   return [(unigramFeatures(word), r) for (r, words) in data for word in words]

def unigramClassifier(train):
   #TODO remove stop-words
   featureSets = unigramFeatureSets(train)
   return nltk.NaiveBayesClassifier.train(featureSets)

# bigram classifier
def bigramFeatures(bigram):
   return {'bigram':bigram}

def bigramFeatureSets(data): #data accepted as (rating, list of words)
   return [(bigramFeatures(bigram), r) for (r, words) in data for bigram in nltk.bigrams(words)]

def bigramClassifier(train):
   featureSets = bigramFeatureSets(train)
   return nltk.NaiveBayesClassifier.train(featureSets)

# paragraph classifier
def paragraphFeatures(paragraph):
   #TODO most freq word, least frequent etc.
   return {'paragraph':paragraph}

def paragraphFeatureSets(data): #data accepted as (rating, list of words)
   return [(paragraphFeatures(words), r) for (r, words) in data]

def paragraphClassifier(train):
   #TODO remove stop-words
   featureSets = paragraphFeatureSets(train)
   return nltk.NaiveBayesClassifier.train(featureSets)
