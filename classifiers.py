import nltk

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

   @staticmethod
   def featureSets(data): #data accepted as (rating, list of words)
      return [(ParagraphClassifier.features(words), r) for (r, words) in data]

   @staticmethod
   def features(paragraph):
      #TODO most freq word, least frequent etc.
      return {'first word':paragraph[0]}

