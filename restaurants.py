import random, nltk
import os
import re

#Review Data Structure items
Rauthor = "REVIEWER"
Rname = "NAME"
Raddress = "ADDRESS"
Rcity = "CITY"
Rratings = "RATINGS"
Rparagraphs = "PARAGRAPHS"
Rfname = "FILENAME"

#Number of folds
folds = 4

#Input Data
dataInputs = Rauthor, Rname, Raddress, Rcity
ratingInputs = "FOOD", "SERVICE", "VENUE", "RATING", "OVERALL"
paragraphInput = "WRITTEN REVIEW"

testData = "./test/"
trainData = "./training/"

adjectives = ['JJ','JJR','JJS']
adverbs = ['WRB','RB','RBR','RBS']

def main():
   reviews = splitReviews(getTrainData())

   

def splitReviews(reviews):
   random.shuffle(reviews)
   size = len(reviews) / folds
   splitRevs = []
   for i in range(folds - 1) :
      splitRevs.append(reviews[i * size : size * (i + 1)])
   splitRevs.append(reviews[size * (folds - 1):])


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
   return {'adjective':POS_word(pos,adjectives), 'adverb':POS_word(pos,adverbs)} #'unigram': pos[0], 'pos': pos[1], 

def get_featureSets(data):
   #featureSets = [(trigramFeatures(tri), r) for (r, p) in data for tri in nltk.trigrams(nltk.pos_tag(nltk.word_tokenize(p)))]
   featureSets = [(langFeature(pos), r) for (r, p) in data for pos in nltk.pos_tag(nltk.word_tokenize(p)) if len(pos[0]) > 3]
   return featureSets

def run_tests(data):
   featureSets = get_featureSets(data) # data format: (rating, paragraph)
   random.shuffle(featureSets) # Shuffle the feature sets

   split = len(featureSets)*4/5
   train, test = featureSets[:split], featureSets[split:]

   classifier = nltk.NaiveBayesClassifier.train(train)
   
   print "Accuracy: ", nltk.classify.accuracy(classifier,test)
   classifier.show_most_informative_features(20)

def classify_input(data):
   paragraph = raw_input('Enter text: ') # Get input review paragraph
   classifier = nltk.NaiveBayesClassifier.train(get_featureSets(data))
   feature_set = [trigramFeature(tri) for tri in nltk.trigrams(nltk.word_tokenize(paragraph))]
   print "Rating: ", nltk.classifier.classify(classifier, feature_set)

def getTrainData() :
   reviews = []
   for fname in os.listdir(trainData):
      if fname.endswith('.html') :
         reviews.append(parseReview(trainData, fname))

   return reviews

def getAllParagraphs(reviews) :
   paras = []
   for review in reviews :
      for i in range(len(review[Rparagraphs])) :
         paras.append( (review[Rratings][i], review[Rparagraphs][i]) )

   return paras

def getAllAuthors(reviews) :
   authors = []
   for review in reviews :
      authors.append( (review[Rauthor], review[Rparagraphs]) )

   print authors
   return authors


def parseReview(path, fname) :

   review = {}
   rating = []
   paras = []

   fp = open(path + fname, 'r')
   readingParasFlag = False
   for line in fp.readlines() :
      line = stripHTML(line).strip()
      temp = [split.strip() for split in line.split(":", 1)]
      if readingParasFlag and len(line) > 0 :
         paras.append(nltk.pos_tag(nltk.word_tokenize(line)))
      elif temp[0] == paragraphInput :
         readingParasFlag = True
      elif temp[0] in dataInputs :
         review[temp[0]] = temp[1]
      elif temp[0] in ratingInputs :
         rating.append(int(temp[1]))

   fp.close()

   review[Rfname] = fname
   review[Rparagraphs] = paras
   review[Rratings] = rating

   checkReview(review)

   return review

def checkReview(review) :
   if len(review.keys()) != 7 :
      print "   Expected 7 keys, got " + str(len(review.keys()))
   if len(review[Rratings]) != 4 :
      print "   Expected 4 ratings, got " + str(len(review[Rratings]))
   if len(review[Rparagraphs]) != 4 :
      print "   Expected 4 paragraphs, got " + str(len(review[Rparagraphs]))

def printReview(review) :
   #Print metadata
   for key, value in review.items() :
      if key != paragraphs:
         print "'" + key + "' : '" + str(value) + "'"
   #Print paragraphs      
   for line in review[paragraphs] :
      print "-----"
      print line
   print "-----"
      

def stripHTML(text) :
   newText = re.sub("<.*?>", " ", text)
   newText = re.sub("\xc2\xa0", " ", newText) #remove &nbsp;
   return newText

if __name__ == '__main__':
   main()