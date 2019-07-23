import nltk
nltk.download('treebank')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

### build up a model to perform sentence segmentation
# convert data into form that can do extracting features
model_sent = nltk.corpus.treebank_raw.sents()
tokens = []
boundaries = set()
offset = 0
for model_sent in nltk.corpus.treebank_raw.sents():
    tokens.extend(model_sent)
    offset += len(model_sent)
    boundaries.add(offset-1)

# function to extract features
def punct_features(tokens, i):
    return {'next_word_cap': tokens[i+1][0].isupper(),
            'prev_word': tokens[i-1].lower(),
            'punct': tokens[i],
            'prev_word_is_one_char': len(tokens[i-1])==1}

# create featuresets
featuresets = [(punct_features(tokens, i), (i in boundaries))
               for i in range(1, len(tokens)-1)
               if tokens[i] in '.?!']

# build punctuation classifier model
size = int(len(featuresets)*0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
#print('model accuracy:', nltk.classify.accuracy(classifier, test_set))

# function for sentences segmentation using model classifier
def segment_sents(words):
    start = 0
    sents = []
    for i in range(1, len(words) - 1):
        if words[i] in '.?!':
            words_features = (punct_features(words, i))
            if classifier.classify(words_features) == True:
                sents.append(words[start:i+1])
                start = i + 1
    if start < len(words):
        sents.append(words[start:])
    return (sents)