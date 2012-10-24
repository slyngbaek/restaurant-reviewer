import nltk, featureutils, math
from featureutils import *

class SentenceClassifier(object):
   """Sentence Classifier - accepts data in (rating, list of words) format"""
   def __init__(self, data):
      self.uclassifier = UnigramClassifier(data)
      featureSets = self.featureSets(data)
      self.classifier = nltk.NaiveBayesClassifier.train(featureSets)

   def classify(self, sentence):
      return self.classifier.classify(self.features(sentence))

   def probClassify(self, sentence):
      return self.classifier.prob_classify(self.features(sentence))

   def classifyParagraphWithSkew(self, p):
      rating = 0
      total = 0
      sentence = []
      counts = {}
      for word in p:
         sentence.append(word)
         if isEndOfSentence(word):
            prob = self.probClassify(sentence)
            rating += ratingFromProb(prob,[3,1,1,1,2])
            total += 1
            sentence = []
      if total < 1:
         return 4
      return float(rating)/total

   def classifyParagraphWithWeight(self, p):
      rating = 0
      total = 0
      sentence = []
      counts = {}
      for word in p:
         sentence.append(word)
         if isEndOfSentence(word):
            prob = self.probClassify(sentence)
            r = prob.max()
            w = prob.prob(r)
            rating += r*w
            total += w
            sentence = []
      if total < 1:
         return 4
      return float(rating)/total

   def classifyParagraph(self, p):
      rating = 0
      total = 0
      snum = 0
      sentence = []
      counts = {}
      for word in p:
         sentence.append(word)
         if isEndOfSentence(word):
            w = 1
            if snum == 0:
               w = 2
            snum += 1
            rating += self.classify(sentence)*w
            total += w
            sentence = []
      if total < 1:
         return 4
      return float(rating)/total

   def most_informative_features(self, n=20):
      self.uclassifier.most_informative_features(n)
      return self.classifier.show_most_informative_features(n)

   def accuracy(self, test):
      return nltk.classify.accuracy(self.classifier, self.featureSets(test))

   def featureSets(self, data): #data accepted as (rating, list of words)
      sentences = []
      for (r, words) in data:
         sentence = []
         for word in words:
            sentence.append(word)
            if isEndOfSentence(word):
               sentences.append(( self.features(sentence), r ))
               sentence = []

      return sentences

   def features(self, sentence):
      fs = {}

      taggedwords = nltk.pos_tag(sentence)
      posWords = [stem(word.lower()) for (word, tag) in taggedwords if tag in goodTags]
      # simpleWords = [stem(word.lower()) for word in sentence if not isStopWord(word) and not isPunctuation(word)]
      sentiments = [sentiment(word) for word in sentence]
      lsentiWords = [stem(word.lower()) for word in sentence if sentiment(word) < 0]
      hsentiWords = [stem(word.lower()) for word in sentence if sentiment(word) > 0]

      fs['words'] = int(len(sentence)/4 + .5)
      fs['unigram rating'] = int(4*self.uclassifier.classifyParagraph(sentence)+0.5)

      addKeysToDict(posWords,fs)
      return fs

class CharacterNgramClassifier(object):
   """Character Ngram Classifier - accepts data in the (rating, list of words) format"""
   def __init__(self, authors):
      self.authorProfiles = CharacterNgramClassifier.__getAuthorProfiles(authors)
      self.featureSets = CharacterNgramClassifier.featureSets(authors)

   def classify(self, review):
      trigrams = []
      for p in review:
         trigrams.extend(nltk.trigrams(re.sub("[^a-z]", "", "".join(p).lower())))
      trigrams = ["".join(t) for t in trigrams]

      test_profile = CharacterNgramClassifier.__getNormalizedTrigramFreq(trigrams)

      result_dict = {}
      for name, profile in self.authorProfiles.iteritems():
         dissimilarity = 0
         for tri in test_profile:
            # Dissimilarity Function
            fa = test_profile[tri]
            fb = profile.get(tri, 0)
            dissimilarity += math.pow(((2 * (fa - fb)) / (fa + fb)), 2) / (4 * len(trigrams))
         similarity = 1 - (dissimilarity * 1.8)
         result_dict[name] = similarity
      return result_dict

   def classifyParagraph(self, p):
      return self.classify(p)

   def most_informative_features(self, n=20):
      return self.classifier.show_most_informative_features(n)

   def accuracy(self, test):
      return nltk.classify.accuracy(self.classifier, CharacterNgramClassifier.featureSets(test))

   @staticmethod
   def __getAuthorProfiles(authors):
      author_list = {}
      for author in authors:
         tris = []
         for p in authors[author]:
            tris.extend(nltk.trigrams(re.sub("[^a-z]", "", "".join(p).lower())))
         tris = ["".join(t) for t in tris]
         author_list[author] = tris

      return {k: CharacterNgramClassifier.__getNormalizedTrigramFreq(v) 
                  for (k, v) in author_list.iteritems()}

   @staticmethod
   def __getNormalizedTrigramFreq(trigrams):
      triFreq = nltk.FreqDist(trigrams)

      tri_dict = {}
      for t in triFreq.keys()[:300]:
         tri_dict[t] = triFreq.freq(t)
      return tri_dict

   @staticmethod
   def featureSets(authors): #data returned as (rating, list of words)
      authorProfiles = CharacterNgramClassifier.__getAuthorProfiles(authors)
      return [(v, k) for (k, v) in authorProfiles.iteritems()]


class UnigramClassifier(object):
   """Unigram Classifier - accepts data in the (rating, list of words) format"""
   """Used by sentence classifier """
   def __init__(self, data):
      self.featureSets = UnigramClassifier.featureSets(data)
      self.classifier = nltk.NaiveBayesClassifier.train(self.featureSets)

   def classify(self, word):
      return self.classifier.classify(UnigramClassifier.features(word))

   def probClassify(self, word):
      return self.classifier.prob_classify(UnigramClassifier.features(word))

   def classifyWithCustomWeights(self, p):
      rating = 0
      count = 0
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
      return float(rating)/count

   def classifyWithWeights(self, p):
      rating = 0
      total = 0
      for word in p:
         if not isStopWord(word) and not isPunctuation(word):
            prob = self.probClassify(word)
            r = prob.max()
            w = prob.prob(r)
            rating += r*w
            total += w

      if total < 1:
         return 4
      return float(rating)/total

   def classifyWithHistogram(self, p):
      rating = 0
      total = 0
      count = 0
      counts = {}

      for word in p:
         if not isStopWord(word) and not isPunctuation(word):
            prob = self.probClassify(word)
            r = prob.max()
            w = prob.prob(r)
            rating += r*w
            total += w

      if total < 1:
         return 4
      return float(rating)/total

   def classifyParagraph(self, p):
      rating = 0
      total = 0

      for word in p:
         if not isStopWord(word) and not isPunctuation(word):
            rating += self.classify(word)
            total += 1
    
      if total < 1:
         return 4
      return float(rating)/total

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
      return UnigramClassifier.simpleFeatureSets(data)
      fs = UnigramClassifier.leastFreqFeatureSets(data)
      fs.extend(UnigramClassifier.mostFreqFeatureSets(data,4))
      return fs

   @staticmethod
   def posFeatureSets(data): #data accepted as (rating, list of words)
      fs = [];
      for (r, words) in data:
         taggedWords = nltk.pos_tag(words)
         fs.extend([(UnigramClassifier.features(stem(word.lower())), r) for (word, tag) in taggedWords
                                                    if tag in goodTags])
      return fs

   @staticmethod
   def leastFreqFeatureSets(data, n=10): #data accepted as (rating, list of words)
      fs = []
      for (r, words) in data:
         words = [word.lower() for word in words if not isStopWord(word) and not isPunctuation(word)]
         words = leastFrequent(words, n)
         fs.extend([(UnigramClassifier.features(word), r) for word in words])
      return fs

   @staticmethod
   def mostFreqFeatureSets(data, n=10): #data accepted as (rating, list of words)
      fs = []
      for (r, words) in data:
         words = [word.lower() for word in words if not isStopWord(word) and not isPunctuation(word)]
         words = mostFrequent(words, n)
         fs.extend([(UnigramClassifier.features(word), r) for word in words])
      return fs

   @staticmethod
   def simpleFeatureSets(data): #data accepted as (rating, list of words)
      return [(UnigramClassifier.features(word.lower()), r) for (r, words) in data for word in words
                                                    if not isStopWord(word) and not isPunctuation(word)]

   @staticmethod
   def features(word):
      return {'word':word, 'sentiment':int(sentiment(word)/4), 'wordlen':int(len(word)/5 + .5)}



""" Failed Attempts of classifiers (From here down) """


class BigramClassifier(object):
   """Bigram Classifier - accepts data in the (rating, list of words) format"""
   def __init__(self, data):
      self.featureSets = BigramClassifier.featureSets(data)
      self.classifier = nltk.NaiveBayesClassifier.train(self.featureSets)

   def classify(self, bigram):
      return self.classifier.classify(BigramClassifier.features(bigram))

   def probClassify(self, bigram):
      return self.classifier.prob_classify(BigramClassifier.features(bigram))

   def classifyParagraph(self, p):
      nicewords = [word.lower() for word in p if not isStopWord(word) and not isPunctuation(word)]
      bigrams = nltk.bigrams(nicewords)
      rating = 0
      total = 0
      for bigram in bigrams:
         prob = self.probClassify(bigram)
         r = prob.max()
         w = prob.prob(r)
         rating += r*w
         total += w
      return float(rating)/total

   def most_informative_features(self, n=20):
      return self.classifier.show_most_informative_features(n)

   def accuracy(self, test):
      return nltk.classify.accuracy(self.classifier, BigramClassifier.featureSets(test))

   @staticmethod
   def featureSets(data): #data accepted as (rating, list of words)
      fs = [] 
      for (r, words) in data:
         nicewords = [word.lower() for word in words if not isStopWord(word) and not isPunctuation(word)]
         for bigram in nltk.bigrams(nicewords):
            fs.append((BigramClassifier.features(bigram),r))

      return fs

      return [(BigramClassifier.features(bigram), r)  for bigram in nltk.bigrams(words)]

   @staticmethod
   def features(bigram):
      return {'sentiment':sentiment(bigram[0])}


class ParagraphClassifier(object):
   """Paragraph Classifier - accepts data in the (rating, list of words) format"""
   def __init__(self, data):
      self.sentenceClassifier = SentenceClassifier(data)
      featureSets = self.featureSets(data)
      self.classifier = nltk.MaxentClassifier.train(featureSets)

   def classify(self, paragraph):
      return self.classifier.classify(self.features(paragraph))

   def classifyParagraph(self, p):
      return self.classify(p)

   def most_informative_features(self, n=20):
      self.sentenceClassifier.most_informative_features(n)
      return self.classifier.show_most_informative_features(n)

   def accuracy(self, test):
      return nltk.classify.accuracy(self.classifier, self.featureSets(test))

   def featureSets(self, data): #data accepted as (rating, list of words)
      return [(self.features(words), r) for (r, words) in data]

   def features(self, paragraph):
      fs = {}

      fs['sentences'] = int(numberOfSentences(paragraph)/2)
      fs['sentenceRating'] = int(self.sentenceClassifier.classifyParagraph(paragraph)+.5)

      return fs

      taggedlist = nltk.pos_tag(paragraph)
      adjectives = [w for (w,tag) in taggedlist if tag in adjectiveTags or tag in adverbTags]
      
      return {'frequent adjectives': ' '.join(mostFrequent(adjectives,2)), 
              'infrequent adjective': leastFrequent(adjectives)[0], 
              'least frequent':leastFrequent(words)[0],
              'most frequent':mostFrequent(words)[0] }
      
      fs = {}
      for (w, tag) in taggedlist:
         if tag in adjectiveTags:
            fs[w] = 1
      return fs #{'first word':paragraph[0]}

class BrettClassifier(object):
   """ Classifier - accepts data in the (rating, list of words) format"""
   def __init__(self, data):
      self.featureSets = self.featureSets(data)
      #self.classifier = nltk.NaiveBayesClassifier.train(self.featureSets)
      #self.classifier = nltk.DecisionTreeClassifier.train(self.featureSets)
      self.classifier = nltk.MaxentClassifier.train(self.featureSets, trace = 1)

   def classifyParagraph(self, p):
      return self.classifier.classify(self.features(p))

   def featureSets(self, data): 
      #data accepted as (rating, list of words)
      return [(self.features(p), r) for (r, p) in data]

   def features(self, paragraph) :
      features = self.unigramsPOS(paragraph)
      return features

   def unigrams(self, paragraph):
      """ Creates a feature set for one paragraph"""
      freq = nltk.FreqDist([word.lower() for word in paragraph if not isStopWord(word)])
      return {key : True for key in freq.keys()}

   def unigramsPOS(self, paragraph):
      selectTags = ['JJ','JJR','JJS', 'WRB','RB','RBR','RBS']
      words = nltk.pos_tag(paragraph)
      freq = nltk.FreqDist([word.lower() for word, POS in words if POS in selectTags ])
      return {key : True for key in freq.keys()}

   def bigrams(self, paragraph):
      bigrams = nltk.util.bigrams(paragraph)
      freq = nltk.FreqDist(bigrams)
      return {key : freq.freq(key) for key in freq.keys() if key not in freq.hapaxes()}
