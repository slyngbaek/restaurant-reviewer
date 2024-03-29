import re, nltk, operator
from string import atof

adjectiveTags = ['JJ','JJR','JJS']
adverbTags = ['WRB','RB','RBR','RBS']
comparativeTags = ['RBR','JJR']

goodTags = ['RBR','JJS','CC','JJR'] #':'
goodWords = ['good','great','awesome','bad','less','more','better', 'food', 'waitress', 'grill', 'pretty', 'always', 'nice', 'quality', 'rather', 'quick', 'lot']



def ratingFromProb(p,skew=[1,1,1,1,1]):
   ps = {}
   for i in range(1,6):
      ps[i] = p.prob(i)*skew[i-1]
   return max(ps.iteritems(), key=operator.itemgetter(1))[0]

def buildSenti():
   f = open('senti_wordnet.txt','r')
   raw = f.read()

   r = re.compile('(\S*)\s*(\S*)\s*(\S*)\s')
   results = r.findall(raw)

   senti = dict((res[2], (atof(res[0]), atof(res[1]))) for res in results)
   f.close()
   return senti

senti_wordnet = buildSenti()

def sentiment(word):
   if word in senti_wordnet:
      senti = senti_wordnet[word][0] - senti_wordnet[word][1]
      return int(senti*10)
      if senti > .6:
         return 2
      elif senti > .2:
         return 1
      elif senti > -.2:
         return 0
      elif senti > -.6:
         return -1
      else:
         return -2
   return 0

def mostFrequent(words,n=1):
   if len(words) < 2:
      return words
   
   counts = {}
   for word in words:
      if word in counts:
         counts[word] += 1
      else:
         counts[word] = 1
   
   words = sorted(counts, key = counts.get, reverse = True)
   if len(words) < n:
      n = len(words)
   return words[:n]

def leastFrequent(words,n=1):
   if len(words) < 2:
      return words
   
   counts = {}
   for word in words:
      if word in counts:
         counts[word] += 1
      else:
         counts[word] = 1
   
   words = sorted(counts, key = counts.get)
   if len(words) < n:
      n = len(words)
   return words[:n]

def stem(word):
   if len(word) < 3:
      return word
   if word[-3:] in ['ing']:
      return word[:-3]
   elif word[-2:] in ['ed','er']:
      return word[:-2]
   elif word[-1:] in ['s','y','.']:
      return word[:-1]
   return word

def isStopWord(word):
   return word.lower() in nltk.corpus.stopwords.words('english')

def isPunctuation(word):
   return word[0] in ".,/?':;!$%()-"

def numberOfSentences(p):
   count = 0
   for word in p:
      if isEndOfSentence(word):
         count = count + 1
   return count

def averageSentenceLength(p):
   count = 0
   curLen = 0
   totalLen = 0 
   for word in p:
      curLen += 1
      if isEndOfSentence(word):
         totalLen += curLen
         count += 1
         curLen = 0
   return float(totalLen)/count

def isEndOfSentence(word):
   return word[0] in ".?!" or word[-1:] in ".?!"

def addKeysToDict(keys, d, value=True):
   for key in keys:
      d[key] = value
