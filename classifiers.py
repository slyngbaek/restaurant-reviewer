import nltk, collections

adjectiveTags = ['JJ','JJR','JJS']
adverbTags = ['WRB','RB','RBR','RBS']
comparativeTags = ['RBR','JJR']

goodTags = [':','RBR','JJS','CC']

def mostFrequent(words,n=1):
   if len(words) < 2:
      return words
   
   counts = {}
   for word in words:
      if word in counts:
         counts[word] += 1
      else:
         counts[word] = 1
   
   words = sorted(counts, key = counts.get, reverse = True)
   if len(words) < n:
      n = len(words)
   return words[:n]

def leastFrequent(words,n=1):
   if len(words) < 2:
      return words
   
   counts = {}
   for word in words:
      if word in counts:
         counts[word] += 1
      else:
         counts[word] = 1
   
   words = sorted(counts, key = counts.get)
   if len(words) < n:
      n = len(words)
   return words[:n]

def isStopWord(word):
   return word.lower() in nltk.corpus.stopwords.words('english')

def isPunctuation(word):
   return word[0] in ".,/?':;!$%"

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

   def most_informative_features(self, n=20):
      return self.classifier.show_most_informative_features(n)

   def accuracy(self, test):
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
         print tag+': ', nltk.classify.accuracy(self.classifier, UnigramClassifier.featureSets(POS[tag]))
      #return nltk.classify.accuracy(self.classifier, UnigramClassifier.featureSets(test))

   @staticmethod
   def featureSets(data): #data accepted as (rating, list of words)
      return [(UnigramClassifier.features(word.lower()), r) for (r, words) in data for word in words
                                                    if not isStopWord(word) and not isPunctuation(word)]

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
      words = [word for word in paragraph if not isStopWord(word) and not isPunctuation(word)]
      # return {'least frequent': min(set(words), key=words.count)}

      taggedlist = nltk.pos_tag(paragraph)
      adjectives = [w for (w,tag) in taggedlist if tag in adjectiveTags or tag in adverbTags]
      # mAds = leastFrequent(adjectives,10)
      # fs = {}
      # for i in range(len(mAds)):
      #    fs[mAds[i]] = 1
      # return fs
      return {'frequent adjectives': ' '.join(mostFrequent(adjectives,2)), 
              'infrequent adjective': leastFrequent(adjectives)[0], 
              'least frequent':leastFrequent(words)[0],
              'most frequent':mostFrequent(words)[0] }
      # fs = {}
      # for (w, tag) in taggedlist:
      #    if tag in adjectives:
      #       fs[w] = 1
      # return fs #{'first word':paragraph[0]}
