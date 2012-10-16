import random, nltk, parser, classifiers
from parser import *
from classifiers import *

#Number of folds
folds = 4

#adjectives = ['JJ','JJR','JJS']
#adverbs = ['WRB','RB','RBR','RBS']

def main():
   reviews = splitReviews(getTrainData())

   for i in range(folds):
      test = reviews[i]
      train = []
      for j in range(folds):
         if i != j:
            train.extend(reviews[j])
      classifyParagraphs(getAllParagraphs(test), getAllParagraphs(train))

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
   classifier = UnigramClassifier(trainingData) #BigramClassifier(trainingData)  or ParagraphClassifier(trainingData)
   total = 0

   for (r, p) in testData:
      rating = classifier.classifyParagraph(p)
      print 'Rating: ', rating
      total += rating
      #TODO calculate root mean square error

if __name__ == '__main__':
   main()