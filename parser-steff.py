import nltk, os, re

#Review Data Structure items
Rauthor = "REVIEWER"
Rname = "NAME"
Raddress = "ADDRESS"
Rcity = "CITY"
Rratings = "RATINGS"
Rparagraphs = "PARAGRAPHS"
Rfname = "FILENAME"

#Input Data
dataInputs = Rauthor, Rname, Raddress, Rcity
ratingInputs = "FOOD", "SERVICE", "VENUE", "RATING", "OVERALL"
paragraphInput = "WRITTEN REVIEW"

testData = "./test/"
trainData = "./training/"

class Review(object):
   """A user review"""
   def __init__(self, reviewer, name, address, city, ratings, paragraphs, filename):
      super(Review, self).__init__()
      self.reviewer = reviewer
      self.name = name
      self.address = address
      self.city = city
      self.ratings = ratings
      self.paragraphs = paragraphs
      self.filename = filename

   # Fix this shit
   def printReview(self):
      #Print metadata
      for key, value in review.items():
         if key != paragraphs:
            print "'" + key + "': '" + str(value) + "'"
      #Print paragraphs      
      for line in review[paragraphs]:
         print "-----"
         print line
      print "-----"

def getTrainData():
   reviews = []
   for fname in os.listdir(trainData):
      if fname.endswith('.html'):
         reviews.append(parseReview(trainData, fname))

   return reviews

def getAllParagraphs(reviews):
   paras = []
   for review in reviews:
      for i in range(len(review[Rparagraphs])):
         paras.append( (review[Rratings][i], review[Rparagraphs][i]) )

   return paras

def getAllAuthors(reviews):
   authors = []
   for review in reviews:
      authors.append( (review[Rauthor], review[Rparagraphs]) )

   print authors
   return authors


def parseReview(path, fname):
   r = Review()
   r.fname = fname
   fp = open(path + fname, 'r')
   readingParasFlag = False
   for line in fp.readlines():
      line = stripHTML(line).strip()
      field = [split.strip() for split in line.split(":", 1)]
      if readingParasFlag and len(line) > 0:
         r.paragraphs.append(nltk.word_tokenize(line)) 
      elif field[0] == paragraphInput:
         readingParasFlag = True
      elif field[0] in dataInputs:
         setattr(r, field[0].lower(), field[1])
      elif field[0] in ratingInputs:
         r.ratings.append(int(field[1]))

   fp.close()
   checkReview(r)
   return r

def checkReview(review):
   if len(review.keys()) != 7:
      print "   Expected 7 keys, got " + str(len(review.keys()))
   if len(review[Rratings]) != 4:
      print "   Expected 4 ratings, got " + str(len(review[Rratings]))
   if len(review[Rparagraphs]) != 4:
      print "   Expected 4 paragraphs, got " + str(len(review[Rparagraphs]))

def stripHTML(text):
   # nltk has a clean HTML method
   # nltk.util.clean_html(html)
   newText = re.sub("<.*?>", " ", text)
   newText = re.sub("\xc2\xa0", " ", newText) #remove &nbsp;
   return newText