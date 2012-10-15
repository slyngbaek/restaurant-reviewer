import operator,nltk

def langFeatures(paragraph):
   words = paragraph.split()
   return {'first_word':words[0]}

def get_featureSets(data):
   return featureSets = [(langFeatures(p),rating) for (rating,p) in data]

def trained_classifier(featureSets):  
   return nltk.NaiveBayesClassifier.train(train)

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
   
   print "Rating: ", classifier.classify(langFeatures(text))


