import nltk, os, re

testData = "./test/"
trainData = "./training/"

class Review(object):
   """A user review"""
   """def __init__(self, reviewer, name, address, city, ratings, paragraphs, filename):
      super(Review, self).__init__()
      self.reviewer = reviewer
      self.name = name
      self.address = address
      self.city = city
      self.ratings = ratings
      self.paragraphs = paragraphs
      self.filename = filename"""
   def __init__(self):
      super(Review, self).__init__()
      self.ratings = []
      self.paragraphs = []


   # Fix this shit
   def printReview(self):
      #Print metadata
      print "Filename   : " + self.filename
      print "Name       : " + self.name
      print "Address    : " + self.address
      print "City       : " + self.city
      print "Reviewer   : " + self.reviewer
      print "Ratings    : " + str(self.ratings)
      #Print paragraphs      
      for line in self.paragraphs:
         print "-----"
         print line
      print "-----"

def getTrainData():
   reviews = []
   for fname in os.listdir(trainData):
      if fname.endswith('.html'):
         #reviews.extend(parseReview(trainData, fname))

         curr = parseReview(trainData, fname)
         for r in curr: 
            r.printReview()
            print ""
         reviews.extend(curr)

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
   #Input Data
   dataInputs = "REVIEWER", "NAME", "ADDRESS", "CITY"
   ratingInputs = "FOOD", "SERVICE", "VENUE", "RATING", "OVERALL"
   paragraphInput = "WRITTEN REVIEW"

   reviews = []
   fp = open(path + fname, 'r')

   r = Review()
   readingParasFlag = False

   lines = stripHTML(fp.read())
   for line in lines:
      line = line.strip()
      field = [split.strip() for split in line.split(":", 1)]
      if readingParasFlag and len(line) > 0:
         r.paragraphs.append(nltk.word_tokenize(line))
         if len(r.paragraphs) == 4:
            #Next Review
            readingParasFlag = False
            reviews.append(r)
            r = Review()
      elif field[0] == paragraphInput:
         readingParasFlag = True
      elif field[0] in dataInputs:
         setattr(r, field[0].lower(), field[1])
      elif field[0] in ratingInputs:
         r.ratings.append(int(field[1]))
   
   #Set Filenames (add -1, -2 etc)
   if len(reviews) == 1:
      reviews[0].filename = fname
   else:
      for i, r in enumerate(reviews):
         r.filename = fname + "-" + str(i + 1)


   fp.close()
   #checkReview(r)
   return reviews

def checkReview(review):
   if len(review.keys()) != 7:
      print "   Expected 7 keys, got " + str(len(review.keys()))
   if len(review[Rratings]) != 4:
      print "   Expected 4 ratings, got " + str(len(review[Rratings]))
   if len(review[Rparagraphs]) != 4:
      print "   Expected 4 paragraphs, got " + str(len(review[Rparagraphs]))

def stripHTML(text):
   minlines = 12

   #remove &nbsp;
   text = re.sub("\xc2\xa0", " ", text) 

   #Try just removing them
   newText = re.sub("<.*?>", " ", text)
   lines = newText.splitlines()
   if len(lines) >= minlines :
      return lines

   #Try making double <br /> into newlines
   newText = re.sub("(<\s*br\s*/>){2}", "\n", text)
   newText = re.sub("<.*?>", " ", newText)
   lines = newText.splitlines()
   if len(lines) >= minlines :
      return lines

   #Make them all newlines instead.
   newText = re.sub("<.*?>", "\n", text)
   lines = newText.splitlines()
   
   return lines

if __name__ == '__main__':
   main()

















