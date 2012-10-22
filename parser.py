import nltk, os, re

testDataDir = "./test/"
trainDataDir = "./training/"

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

def getDataFromDir(directory):
   reviews = []
   count = 0
   for fname in os.listdir(directory):
      if fname.endswith('.html'):
         count += 1
         reviews.extend(parseReview(directory, fname))

   return reviews, count

def getData():
   (train, trainDocs) = getDataFromDir(trainDataDir)
   (tests, testDocs) = getDataFromDir(testDataDir)

   print "   " + str(trainDocs) + " training documents found."
   print "   " + str(len(train)) + " reviews digitized."
   print "   " + str(testDocs)  + " test documents found."

   return (train, tests)

def getAllParagraphs(reviews):
   paras = []
   for review in reviews:
      for i in range(len(review.paragraphs)):
         paras.append( (review.ratings[i], review.paragraphs[i]) )

   return paras

def getAllAuthors(reviews):
   authors = {}
   for r in reviews:
      curr = authors.get(r.reviewer, [])
      curr.extend(r.paragraphs)
      authors[r.reviewer] = curr
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
            #Next Review, Hack
            readingParasFlag = False
            reviews.append(r)
            r = Review()
      elif field[0] == paragraphInput:
         readingParasFlag = True
      elif field[0] in dataInputs:
         setattr(r, field[0].lower(), field[1])
      elif field[0] in ratingInputs:
         try: 
            r.ratings.append(int(field[1]))
         except ValueError:
            r.ratings.append(-1)
   
   #Set Filenames (add -1, -2 etc if needed)
   if len(reviews) == 1:
      reviews[0].filename = fname
   else:
      for i, r in enumerate(reviews):
         r.filename = fname + "-" + str(i + 1)

   fp.close()
   return reviews

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