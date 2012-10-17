import nltk, featureutils
from featureutils import *

class UnigramClassifier(object):
   """Unigram Classifier - accepts data in the (rating, list of words) format"""
   def __init__(self, data):
      self.featureSets = UnigramClassifier.featureSets(data)
      self.classifier = nltk.NaiveBayesClassifier.train(self.featureSets)

   def classify(self, word):
      return self.classifier.classify(UnigramClassifier.features(word))

   def classifyParagraph(self, p):
      rating = 0
      count = 0
      
      if (0):
         taggedWords = nltk.pos_tag(p)
         for (word, tag) in taggedWords:
            if not isStopWord(word) and not isPunctuation(word):
               if (tag in comparativeTags):
                  weight = 10
               elif (tag in adjectiveTags):
                  weight = 2
               elif (tag in adverbTags):
                  weight = 3
               else:
                  weight = 1
               rating += weight*self.classifier.classify(UnigramClassifier.features(word))
               count += weight 
      else:
         for word in p:
            if not isStopWord(word) and not isPunctuation(word):
               rating += 1*self.classifier.classify(UnigramClassifier.features(word))
               count += 1 
      return float(rating)/count

   def most_informative_features(self, n=50):
      return self.classifier.show_most_informative_features(n)

   def accuracyByPOS(self, test):
      POS = {}
      for (r, words) in test:
         taggedwords = nltk.pos_tag(words)
         pos = {}
         for (word, tag) in taggedwords:
            if not tag in pos:
               pos[tag] = [word]
            else:
               pos[tag].append(word)

         for tag in pos:
            if not tag in POS:
               POS[tag] = [(r, pos[tag])]
            else:
               POS[tag].append((r, pos[tag]))

      for tag in POS:
         POS[tag] = nltk.classify.accuracy(self.classifier, UnigramClassifier.featureSets(POS[tag]))

      return POS

   def accuracy(self, test):      
      return nltk.classify.accuracy(self.classifier, UnigramClassifier.featureSets(test))

   @staticmethod
   def featureSets(data): #data accepted as (rating, list of words)
      # fs = [];
      # for (r, words) in data:
      #    taggedWords = nltk.pos_tag(words)
      #    fs.extend([(UnigramClassifier.features(word.lower()), r) for (word, tag) in taggedWords
      #                                               if tag in goodTags])
      # return fs

      fs = []
      for (r, words) in data:
         words = [stem(word.lower()) for word in words if not isStopWord(word) and not isPunctuation(word)]
         words = leastFrequent(words, (len(words)*2)/10)
         fs.extend([(UnigramClassifier.features(word), r) for word in words[:]])
      return fs

      return [(UnigramClassifier.features(stem(word.lower())), r) for (r, words) in data for word in words
                                                    if not isStopWord(word) and not isPunctuation(word)]

   @staticmethod
   def features(word):
      return {'word':word, 'sentiment':sentiment(word)}
     

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

   def most_informative_features(self, n=20):
      return self.classifier.show_most_informative_features(n)

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
      words = [stem(word.lower()) for word in paragraph if not isStopWord(word) and not isPunctuation(word)]
      fs = {}
      for w in words:
         fs[w] = 1
      return fs
      return {'least frequent': min(set(words), key=words.count)}

      taggedlist = nltk.pos_tag(paragraph)
      # adjectives = [w for (w,tag) in taggedlist if tag in adjectiveTags or tag in adverbTags]
      
      # mAds = leastFrequent(adjectives,10)
      # fs = {}
      # for i in range(len(mAds)):
      #    fs[mAds[i]] = 1
      # return fs
      # return {'frequent adjectives': ' '.join(mostFrequent(adjectives,2)), 
      #         'infrequent adjective': leastFrequent(adjectives)[0], 
      #         'least frequent':leastFrequent(words)[0],
      #         'most frequent':mostFrequent(words)[0] }
      
      fs = {}
      for (w, tag) in taggedlist:
         if tag in adjectiveTags:
            fs[w] = 1
      return fs #{'first word':paragraph[0]}

class CharacterNgramClassifier(object):
   """Paragraph Classifier - accepts data in the (rating, list of words) format"""
   def __init__(self, data):
      self.featureSets = CharacterNgramClassifier.featureSets(data)
      self.classifier = nltk.NaiveBayesClassifier.train(self.featureSets)

   def classify(self, paragraph):
      return self.classifier.classify(CharacterNgramClassifier.features(paragraph))

   def classifyParagraph(self, p):
      return self.classify(p)

   def most_informative_features(self, n=20):
      return self.classifier.show_most_informative_features(n)

   def accuracy(self, test):
      return nltk.classify.accuracy(self.classifier, CharacterNgramClassifier.featureSets(test))

   @staticmethod
   def featureSets(data): #data accepted as (rating, list of words)
      return [(CharacterNgramClassifier.features(words), r) for (r, words) in data]

   @staticmethod
   def features(paragraph):
      #for word in paragraph for letter in word
      return {'feature': ' '}
