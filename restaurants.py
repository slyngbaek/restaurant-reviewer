import random, nltk, parser, classifiers
import math
from parser import *
from classifiers import *

#Number of folds
folds = 4

def main():
   print "Starting classifier ..."

   train, tests = getData()

   # tests[2 - 1].reviewer = "Devlin Cronin"
   # tests[3 - 1].reviewer = "Mike Buerli"
   # tests[6 - 1].reviewer = "Brett Armstrong"
   # tests[7 - 1].reviewer = "Steffen Lyngbaek"
   # tests[8 - 1].reviewer = "Ryan Verdon"
   # tests[9 - 1].reviewer = "Andrew Sinclair"

   # tests[2 - 1].ratings = [3,2,3,3]
   # tests[3 - 1].ratings = [4,5,5,4]
   # tests[6 - 1].ratings = [4,5,4,4]
   # tests[7 - 1].ratings = [3,4,3,3]
   # tests[8 - 1].ratings = [5,5,4,4]
   # tests[9 - 1].ratings = [4,4,5,4]

   # tests.pop(4)
   # tests.pop(3)
   # tests.pop(0)

   exercise1(train)
   exercise2(train)
   exercise34(train)
   makePredictions(train, tests)

def exercise1(data):
   print "Exercise 1 validation"
   trainSet = splitReviews(data)
   totalrms = 0.0
   for i in range(folds):
      (test, train) = trainSet[i]
      printValidationSet(i, test)
      totalrms += classifyParagraphs(getAllParagraphs(test), getAllParagraphs(train))
   totalrms = totalrms / 4.0

   print "Exercise 1 Overall RMS Error:", totalrms 

def exercise2(data):
   print "Exercise 2 validation"
   trainSet = splitReviews(data)
   totalrms = 0.0
   for i in range(folds):
      (test, train) = trainSet[i]
      printValidationSet(i, test)
      totalrms += classifyReviews(getOverallParagraphs(test), getOverallParagraphs(train))   
   totalrms = totalrms / 4.0

   print "Exercise 2 Overall RMS Error:", totalrms 

def exercise34(data):
   print "Exercise 3 validation"
   trainSet = splitReviewsByAuthor(data)
   totalrms = 0.0
   labels = [author for author in sorted(getAllAuthors(data).keys())]
   sim_matrix = [[(0, 0.0) ] * len(labels) for i in range(len(labels))]
   for i, (test,train) in enumerate(trainSet):
      printValidationSet(i, test)
      totalrms += classifyAuthors(getAllAuthors(test), getAllAuthors(train), sim_matrix, labels)
   totalrms = totalrms / 4.0

   print "Exercise 3 Overall RMS Error:", totalrms 

   print "Similarity Confusion Matrix: "
   sim_matrix = [[float(total) / count for (count, total) in matrixRow] for matrixRow in sim_matrix]
   for num, author in enumerate(labels):
      print str(num + 1) + ".) " + author
   print ""
   printMatrix(sim_matrix, True)

def makePredictions(train, tests):

#[NAME OF FILE], [PARAGRAPH PREDICTIONS 1-4], [OVERALL RATING PREDICTION], [AUTHOR PREDICTION] [NEWLINE]
   outfp = open("predictions.txt", 'w')


   authorClassifier = CharacterNgramClassifier(getAllAuthors(train))
   paragraphClassifier = BrettClassifier(getAllParagraphs(train))

   for rev in tests:
      paragraph_ratings = []
      for p in rev.paragraphs:
         paragraph_ratings.append(paragraphClassifier.classifyParagraph(p))
      results = authorClassifier.classify(rev.paragraphs)
      guess = [key for key, value in results.items() if value == max(results.values())][0]
      overall_rating = sum([.2 * r for r in paragraph_ratings[:-1]]) + .4 * paragraph_ratings[-1]
      paragraph_ratings = [str(r) for r in paragraph_ratings]
      pred_str = rev.filename + ", " + ', '.join(paragraph_ratings) + ", " + str(overall_rating) + ", " + guess
      print pred_str
      outfp.write(pred_str + "\n")

   outfp.close()

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
   
   # for author in trainingData.keys():
   #    if not author in testData.keys():
   #       del trainingData[author]

   classifier = CharacterNgramClassifier(trainingData)
   correct = 0
   totalerror = 0

   
   for author in testData.keys():
      results = classifier.classify(testData[author])
      guess = [key for key, value in results.items() if value == max(results.values())][0]
      rating = -1
      for num, value in enumerate(sorted(results.values(), reverse=True)):
         if value == results[author]:
            rating = num
            break;
      print "Guess: ",'%.3f' % results[guess],'%26s' % guess, " Actual: ", '%.3f' % results[author], author, rating
      if guess != author:
         totalerror += 1
      if guess == author:
         correct += 1

      #Fill in sim_matrix
      row = labels.index(author)
      for key, value in results.items():
         if key.lower() == 'x':
            break;
         col = labels.index(key)
         count, total = sim_matrix[row][col]
         count += 1 
         total += value
         sim_matrix[row][col] = (count, total)

   rms =  math.sqrt(float(totalerror) / len(testData))
   print "Accuracy:", correct, "/", len(testData), " - ", float(correct)/len(testData)  
   print "RMS Error:", rms

   return rms

def classifyReviews(testData, trainingData):
   """ test/training data is of form (list of ratings, list of paragraphs)"""
   trainingData = [(lr[i], lp[i]) for lr, lp in trainingData for i in range(len(lr))]
   classifier = BrettClassifier(trainingData)

   total = 0.0
   count = [[0] * 5 for i in range(5)]
   correct = 0

   for (lr, lp) in testData:
      ratings = []
      for p in lp:
         rating = classifier.classifyParagraph(p)
         ratings.append(rating)
      overall = .2 * ratings[0] + .2 * ratings[1] + .2 * ratings[2] + .4 * ratings[3]
      if overall == lr[3]:
         correct += 1
      print str(lr) + " : " + str(overall), ratings
      total += float((overall - lr[3]) * (overall - lr[3]))

   rms = math.sqrt(total / float(len(testData)))
   print "RMS error : " + str(rms)
   print "Accuracy  : " + str(float(correct) / len(testData))
   classifier.most_informative_features()
   return rms


def classifyParagraphs(testData, trainingData):
   print 'Training classifier'
   classifier = BrettClassifier(trainingData)
   #classifier = SentenceClassifier(trainingData)

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
      total += float((rating - r) * (rating - r))

   rms = math.sqrt(total / float(len(testData)))
   print "RMS error : " + str(rms)
   print "Accuracy  : " + str(float(correct) / len(testData))
   print "Counts    : "
   printMatrix(count, False)
   return rms

def printMatrix(matrix, isFloat):
   print "%7s" % "", ''.join(['%7s' % (i + 1) for i in range(len(matrix))])
   for i in range(len(matrix)):
      formatString = '%7s'
      if isFloat:
         formatString = '%7.3f'
      print '%7s' % (i + 1), ''.join([formatString % val for val in matrix[i]])

if __name__ == '__main__':
   main()
