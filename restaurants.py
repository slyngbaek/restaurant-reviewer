import random, nltk, parser
from parser import *

#Number of folds
folds = 4

adjectives = ['JJ','JJR','JJS']
adverbs = ['WRB','RB','RBR','RBS']

def main():
   reviews = splitReviews(getTrainData())

   for i in range(folds) : 
      test = reviews[i]
      train = []
      for j in range(folds) :
         if i != j :
            train.extend(reviews[j])
      run_tests(getAllParagraphs(train), getAllParagraphs(test))

   

def splitReviews(reviews):
   random.shuffle(reviews)
   size = len(reviews) / folds
   splitRevs = []
   for i in range(folds - 1) :
      splitRevs.append(reviews[i * size : size * (i + 1)])
   splitRevs.append(reviews[size * (folds - 1):])

   return splitRevs


def dontcare():
   return random.randint(-1000000,100000000)

def POS_word(taggedword, POSs):
   if(taggedword[1] in POSs):
      return taggedword[0]
   else:
      return dontcare()

def POS_tuple(trigram, POSs):
   if(trigram[0][1] in POSs):
      return ' '.join([trigram[0][0],trigram[1][0]])
   else:
      return dontcare()

def trigramFeatures(trigram):
   return {'adjective':POS_word(trigram[0], adjectives), 
           #'adverb':POS_word(trigram[0], adverbs),
           #'adjective_t':  POS_tuple(trigram, adjectives), 
           'adverb_t': POS_tuple(trigram, adverbs)} #'trigram': ' '.join([trigram[0][0],trigram[1][0],trigram[2][0]]), 

def langFeature(pos):
   if usePOSTags :
      return {'adjective':POS_word(pos,adjectives), 'adverb':POS_word(pos,adverbs)} #'unigram': pos[0], 'pos': pos[1], 
   else :
      return {'unigram' : pos[0]}

def get_featureSets(data):
   #featureSets = [(trigramFeatures(tri), r) for (r, p) in data for tri in nltk.trigrams(nltk.pos_tag(nltk.word_tokenize(p)))]
   featureSets = [(langFeature(pos), r) for (r, p) in data for pos in p if len(pos[0]) > 3]
   return featureSets

def run_tests(train, test):

   print get_featureSets(train)[:5]

   classifier = nltk.NaiveBayesClassifier.train(get_featureSets(train))
   
   print "Accuracy: ", nltk.classify.accuracy(classifier, get_featureSets(test))
   classifier.show_most_informative_features(20)

def classify_input(data):
   paragraph = raw_input('Enter text: ') # Get input review paragraph
   classifier = nltk.NaiveBayesClassifier.train(get_featureSets(data))
   feature_set = [trigramFeature(tri) for tri in nltk.trigrams(nltk.word_tokenize(paragraph))]
   print "Rating: ", nltk.classifier.classify(classifier, feature_set)



if __name__ == '__main__':
   main()