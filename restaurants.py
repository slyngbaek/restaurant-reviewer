import random, nltk

def langFeatures(trigram):
   return {'word': trigram[0], 'trigram': trigram}

def get_featureSets(data):
   return [[(langFeatures(tri), r) for tri in nltk.trigrams(nltk.word_tokenize(p))] (r, p) in data]


def run_tests(data):
   featureSets = get_featureSets(data) # data format: (rating, paragraph)
   random.shuffle(featureSets) # Shuffle the feature sets

   split = len(featureSets)*4/5
   train, test = featureSets[:split], featureSets[split:]

   classifier = nltk.NaiveBayesClassifier.train(train)
   
   print "Accuracy: ", nltk.classify.accuracy(classifier,test)
   print classifier.show_most_informative_features(20)

def classify_input(data):
   paragraph = raw_input('Enter text: ') # Get input review paragraph
   classifier = nltk.NaiveBayesClassifier.train(get_featureSets(data))
   feature_set = [langFeatures(tri) for tri in nltk.trigrams(nltk.word_tokenize(paragraph))]
   print "Rating: ", nltk.classifier.classify(classifier, feature_set)

def main():
   pass

if __name__ == '__main__':
   main()