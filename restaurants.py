import random, nltk, parser, classifiers
import math
from parser import *
from classifiers import *

#Number of folds
folds = 4

def main():
   print "Starting classifier ..."
   data = getTrainData()

   doExercise(1, data)

#    classifier = CharacterNgramClassifier(getAllAuthors(data))
#    classifier.classify("""
#       Mama's Meatball presented us with a good meal at an above-average price. The food ordered (a "small" relatively simplistic pasta dish) was prepared well, and had a decent aesthetic. As with many of the finer restaurants, the main dish was supplemented with a complimentary addition; at Mama's Meatball, this addition is bread bites. While the pasta ordered was not by any means of poor quality, it was also not anything particularly special. The additional bread bites, however, were very good (and this is likely the reason that we were supplied with three bowls of them over the course of the meal).
# On the night we visited, Mama's Meatball seemed to be serving more customers than the maximum for which the establishment was prepared. While we were treated very well by the wait staff, the meal also took significantly longer to prepare than would have been expected. Due to this extended wait, the service did not rank as highly as it otherwise would have in our minds. That being said, they were still attentive to our needs. We did not have to wait long for refills on our drinks, and never had to signal a waiter (though we are also not wont to do so under normal circumstances).
# The venue was, overall, fairly nice. It provided a small-restaurant feel, and, despite having no free tables at the time of the meal, the restaurant did not seem overly crowded or boisterous, which was a welcome change. However, the decor seemed to over-exaggerate the Italian nature of the restaurant, which actually subtracted slightly from the feel, making it appear more over-the-top and humorous, rather than appealing. This feeling was not aided by the covering of tables by butcher paper, rather than table cloth. This made an otherwise very fine restaurant appear slightly cheaper. An addition to the venue was a piano player, providing reasonable music for the atmosphere. However, the piano was a digital piano, rather than a grand piano, and the difference in sound is apparent. Additionally, the player seemed to have a heavy right hand, frequently playing high notes in forte when they should have been piano. Finally, the parking for Mama's Meatball is not as accessible as would be preferred - though this is a common issue with downtown restaurants in the city of San Luis Obispo, and Mama's Meatball is better than others.
# The restaurant was a pleasant enough experience. The food was good, if not spectacular, as was the venue. While the service was slow, it was neither rude nor negligent. The most significant problem with Mama's Meatball is that it simply is not a good bargain (though it is not entirely unreasonable, either). While the restaurant is decent, the prices would indicate a much finer experience, and more personalized attention by the wait staff. Were the cost of the meal reduced to more closely compete with many of the surrounding restaurants, the rating for Mama's Meatball would rise significantly.
#  """)

   # for N in range(3, 4) :
   #    doExercise(N, data)

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
   classifyAuthors(getAllAuthors(test), getAllAuthors(train))

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

def classifyAuthors(testData, trainingData):
   print 'Training Classifier'
   classifier = CharacterNgramClassifier(trainingData)

   print 'Accuracy: ', classifier.accuracy(testData)
   classifier.most_informative_features()

def classifyParagraphs(testData, trainingData):
   print 'Training classifier'
   classifier = BrettClassifier(trainingData)
   total = 0.0
   count = [[0] * 5 for i in range(5)]
   correct = 0

   for (r, p) in testData:
      rating = int(classifier.classifyParagraph(p))
      count[r - 1][rating - 1] += 1
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
   printMatrix(range(1,6), count)
   classifier.most_informative_features()


def printMatrix(labels, matrix):
   print '  ' + ' '.join([str(num) for num in labels])
   for i in range(len(matrix)):
      print str(labels[i]) + ' ' + ' '.join([str(num) for num in matrix[i] ])


if __name__ == '__main__':
   main()
