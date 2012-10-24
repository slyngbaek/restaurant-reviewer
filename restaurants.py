import random, nltk, parser, classifiers
import math
from parser import *
from classifiers import *

#Number of folds
folds = 4

def main():
   print "Starting classifier ..."
   data = getTrainData()

   exercise1(data)
   #exercise34(data)

def exercise1(data):
   print "Exercise 1 validation"
   trainSet = splitReviews(data)
   for i in range(folds):
      (test, train) = trainSet[i]
      printValidationSet(i, test)
      classifyParagraphs(getAllParagraphs(test), getAllParagraphs(train))

def exercise2(data):
   print "Exercise 2 validation"

def exercise34(data):
   print "Exercise 3 validation"
   trainSet = splitReviewsByAuthor(data)
   labels = [author for author in sorted(getAllAuthors(data).keys())]
   sim_matrix = [[(0, 0.0) ] * len(labels) for i in range(len(labels))]
   for i in range(folds):
      (test, train) = trainSet[i]
      printValidationSet(i, test)
      classifyAuthors(getAllAuthors(test), getAllAuthors(train), sim_matrix, labels)
   
   print "Similarity Confusion Matrix: "
   sim_matrix = [[float(total) / count for (count, total) in matrixRow] for matrixRow in sim_matrix]
   for num, author in enumerate(labels):
      print str(num + 1) + ".) " + author
   print ""
   printMatrix(sim_matrix, True)


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

   trainSet = []
   for i in range(folds):
      trainSet.append((splitRevs[i],[r for j in range(folds) 
                      for r in splitRevs[j] if i != j]))

   return trainSet

def splitReviewsByAuthor(reviews):
   random.shuffle(reviews)
   splitRevs = [[] for i in range(folds)]

   for review in reviews:
      choice = shortestList([revList for revList in splitRevs 
                   if not isAuthorInList(review.reviewer, revList)])
      choice.append(review)

   trainSet = []
   for i in range(folds):
      trainSet.append((splitRevs[i],[r for j in range(folds) 
                      for r in splitRevs[j] if i != j]))

   return trainSet

def shortestList(reviewLists):
   minLen = min([len(revList) for revList in reviewLists])
   return random.choice([revList for revList in reviewLists if len(revList) == minLen])

def isAuthorInList(author, list):
   reviews = [review for review in list if review.reviewer == author]
   return len(reviews) > 0

def classifyAuthors(testData, trainingData, sim_matrix, labels):
   print 'Training Classifier'
   classifier = CharacterNgramClassifier(trainingData)
   correct = 0
   total = 0

   
   for author in testData.keys():
      results = classifier.classify(testData[author])
      guess = [key for key, value in results.items() if value == max(results.values())][0]
      rating = -1
      for num, value in enumerate(sorted(results.values(), reverse=True)):
         if value == results[author]:
            rating = num
            break;
      print "Guess: ",'%.3f' % results[guess],'%26s' % guess, " Actual: ", '%.3f' % results[author], author, rating
      total += rating
      if guess == author:
         correct += 1

      #Fill in sim_matrix
      row = labels.index(author)
      for key, value in results.items():
         col = labels.index(key)
         count, total = sim_matrix[row][col]
         count += 1 
         total += value
         sim_matrix[row][col] = (count, total)

   print "Accuracy:", correct, "/", len(results), " - ", float(correct)/len(results)  
   print "Avg Error:", float(total) / len(results)

      #classifier.classify([word for p in testData[author] for word in p])
      # count[r - 1][rating - 1] += 1
      # if rating == r:
      #    correct += 1
      #print str(r) + " : " + str(rating)
      #if math.fabs(r - rating) >= 2 :
      #   print ' '.join(p)
      # total += (rating - r) * (rating - r)

   #print 'Accuracy: ', classifier.accuracy(testData)
   #classifier.most_informative_features()

def classifyParagraphs(testData, trainingData):
   print 'Training classifier'
   classifier = SentenceClassifier(trainingData) #BrettClassifier UnigramClassifier SentenceClassifier
   total = 0.0
   count = [[0] * 5 for i in range(5)]
   correct = 0

   for (r, p) in testData:
      rating = classifier.classifyParagraph(p)
      count[r - 1][int(rating+.5) - 1] += 1
      if rating == r:
         correct += 1
      #print str(r) + " : " + str(rating)
      #if math.fabs(r - rating) >= 2 :
      #   print ' '.join(p)
      total += (rating - r) * (rating - r)

   rms = math.sqrt(total / float(len(testData)))
   print "RMS error : " + str(rms)
   print "Accuracy  : " + str(float(correct) / len(testData))
   print "Counts    : "
   printMatrix(count, False)
   classifier.most_informative_features()


def printMatrix(matrix, isFloat):
   print "%7s" % "", ''.join(['%7s' % (i + 1) for i in range(len(matrix))])
   for i in range(len(matrix)):
      formatString = '%7s'
      if isFloat:
         formatString = '%7.3f'
      print '%7s' % (i + 1), ''.join([formatString % val for val in matrix[i]])


if __name__ == '__main__':
   main()
