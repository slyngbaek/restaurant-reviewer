import re, nltk
from string import atof

adjectiveTags = ['JJ','JJR','JJS']
adverbTags = ['WRB','RB','RBR','RBS']
comparativeTags = ['RBR','JJR']

goodTags = ['RBR','JJS','CC','JJR'] #':'
goodWords = ['good','great','awesome','bad','less','more','better', 'food', 'waitress', 'grill', 'pretty', 'always', 'nice', 'quality', 'rather', 'quick', 'lot']

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
         return 'high'
      elif senti > .2:
         return 'medium'
      elif senti > -.2:
         return 'neutral'
      elif senti > -.6:
         return 'low'
      else:
         return 'very-low'
    return 'neutral'

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