import nltk

def langFeatures(trigram):
   return {'word':trigram[0], 'trigram':trigram}

def get_featureSets(data):
   featureSets = []
   for (rating,paragraph) in data:
      for trigram in nltk.trigams(nltk.word_tokenize(paragraph))
         featureSets.append((langFeatures(trigram),rating))
   return featureSets

def run_tests(data):
   featureSets = get_featureSets(data)

   split = len(featureSets)*4/5
   train, test = featureSets[:split], featureSets[split:]

   classifier = nltk.NaiveBayesClassifier.train(train)
   
   print "Accuracy: ", nltk.classify.accuracy(classifier,test)
   print classifier.show_most_informative_features(20)

def classify_input(data):
   text = raw_input('Enter text: ')
   featureSets = get_featureSets(data)
   classifier = nltk.NaiveBayesClassifier.train(featureSets)
   
   for trigram in nltk.trigams(nltk.word_tokenize(text))
      print "Rating: ", classifier.classify(langFeatures(trigram))