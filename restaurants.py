import random, nltk, parser, classifiers
import math
from parser import *
from classifiers import *

#Number of folds
folds = 4

def main():

   print "Starting classifier ..."
   data = getTrainData()


   for N in range(1,2) :
      doExercise(N, data)


def doExercise(N, data):
   print "Exercise " + str(N) + " validation"
   reviews = splitReviews(data)
   for i in range(folds): 
      test = reviews[i]
      train = [r for j in range(folds) for r in reviews[j] if i != j]
      printValidationSet(i, test)
      print "Len: " + str(len(test))
      if N == 1 :
         exercise1(test, train)
      elif N == 2 :
         exercise2(test, train)
      elif N == 3 :
         exercise3(test, train)
      else:
         print "Error! bad exercise"

def exercise1(test, train):

   classifyParagraphs(getAllParagraphs(test), getAllParagraphs(train))

def exercise2(test, train):
   pass

def exercise3(test, train):
   pass

def printValidationSet(i, reviews):
   fnames = [r.filename for r in reviews]
   print "Random Validation Set " + str(i) + ": " + str(fnames)

def splitReviews(reviews):
   random.shuffle(reviews)
   size = len(reviews) / folds
   splitRevs = []
   for i in range(folds - 1) :
      splitRevs.append(reviews[i * size : size * (i + 1)])
   splitRevs.append(reviews[size * (folds - 1):])

   return splitRevs

def classifyParagraphs(testData, trainingData):
   print 'Training classifier'
   classifier = UnigramClassifier(trainingData) #UnigramClassifier(trainingData) #BigramClassifier(trainingData)  or ParagraphClassifier(trainingData)
   total = 0.0

   for (r, p) in testData:
      rating = classifier.classifyParagraph(p)
      ##print str(r) + " : " + str(rating)
      total += (rating - r) * (rating - r)
      #TODO calculate root mean square error

   print 'Accuracy: ', classifier.accuracy(testData)
   print classifier.most_informative_features()
   rms = math.sqrt(total / float(len(testData)))
   print "RMS error: ", rms, '\n'

if __name__ == '__main__':
   main()


























