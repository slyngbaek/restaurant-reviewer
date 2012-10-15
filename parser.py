import nltk, os, re

#Review Data Structure items
Rauthor = "REVIEWER"
Rname = "NAME"
Raddress = "ADDRESS"
Rcity = "CITY"
Rratings = "RATINGS"
Rparagraphs = "PARAGRAPHS"
Rfname = "FILENAME"

#use POS Tagging
usePOSTags = True

#Input Data
dataInputs = Rauthor, Rname, Raddress, Rcity
ratingInputs = "FOOD", "SERVICE", "VENUE", "RATING", "OVERALL"
paragraphInput = "WRITTEN REVIEW"

testData = "./test/"
trainData = "./training/"

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
         if usePOSTags:
            paras.append(nltk.pos_tag(nltk.word_tokenize(line)))
         else :
            paras.append( [ (word, ) for word in nltk.word_tokenize(line)] ) 
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