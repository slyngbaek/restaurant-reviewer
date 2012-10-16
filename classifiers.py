import nltk, collections

def filterStopWords(words):
   return [word.lower() for word in words if not word.lower() in nltk.corpus.stopwords.words('english')]

def filterPunctuation(words):
   return [w for w in words if not w[0] in ".,/?':;!"]

class UnigramClassifier(object):
   """Unigram Classifier - accepts data in the (rating, list of words) format"""
   def __init__(self, data):
      self.featureSets = UnigramClassifier.featureSets(data)
      self.classifier = nltk.NaiveBayesClassifier.train(self.featureSets)

   def classify(self, word):
      return self.classifier.classify(UnigramClassifier.features(word))

   def classifyParagraph(self, p):
      #TODO remove stop-words
      rating = 0
      for word in p:
         rating += self.classifier.classify(UnigramClassifier.features(word))
      return float(rating)/len(p)

   def most_informative_features(self, n=50):
      return self.classifier.most_informative_features(n)

   def accuracy(self, test):
      return nltk.classify.accuracy(self.classifier, UnigramClassifier.featureSets(test))

   @staticmethod
   def featureSets(data): #data accepted as (rating, list of words)
      #TODO remove stop-words
      return [(UnigramClassifier.features(word), r) for (r, words) in data for word in words]

   @staticmethod
   def features(word):
      return {'unigram':word}
     

class BigramClassifier(object):
   """Bigram Classifier - accepts data in the (rating, list of words) format"""
   def __init__(self, data):
      self.featureSets = BigramClassifier.featureSets(data)
      self.classifier = nltk.NaiveBayesClassifier.train(self.featureSets)

   def classify(self, bigram):
      return self.classifier.classify(BigramClassifier.features(bigram))

   def classifyParagraph(self, p):
      bigrams = nltk.bigrams(p)
      rating = 0
      for bigram in bigrams:
         rating += self.classifier.classify(BigramClassifier.features(bigram))
      return float(rating)/len(bigrams)

   def most_informative_features(self, n=50):
      return self.classifier.most_informative_features(n)

   def accuracy(self, test):
      return nltk.classify.accuracy(self.classifier, BigramClassifier.featureSets(test))

   @staticmethod
   def featureSets(data): #data accepted as (rating, list of words)
      return [(BigramClassifier.features(bigram), r) for (r, words) in data for bigram in nltk.bigrams(words)]

   @staticmethod
   def features(bigram):
      return {'bigram':bigram}


class ParagraphClassifier(object):
   """Paragraph Classifier - accepts data in the (rating, list of words) format"""
   def __init__(self, data):
      self.featureSets = ParagraphClassifier.featureSets(data)
      self.classifier = nltk.NaiveBayesClassifier.train(self.featureSets)

   def classify(self, paragraph):
      return self.classifier.classify(ParagraphClassifier.features(paragraph))

   def classifyParagraph(self, p):
      return self.classify(p)

   def most_informative_features(self, n=20):
      return self.classifier.show_most_informative_features(n)

   def accuracy(self, test):
      return nltk.classify.accuracy(self.classifier, ParagraphClassifier.featureSets(test))

   @staticmethod
   def featureSets(data): #data accepted as (rating, list of words)
      return [(ParagraphClassifier.features(words), r) for (r, words) in data]

   @staticmethod
   def features(paragraph):
      #TODO most freq word, least frequent etc.
      words = filterStopWords(paragraph)
      words = filterPunctuation(words)
      return {'least frequent': min(set(words), key=words.count)}
      
      # fs = {}
      # for w in words:
      #    fs[w] = w
      # return fs #{'first word':paragraph[0]}

