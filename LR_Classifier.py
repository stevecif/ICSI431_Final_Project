import json
import pandas as pd
from itertools import chain
from sklearn.linear_model import LogisticRegression
import pickle

stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]

#Collect positive tweets
positiveTweets = []
positiveTweets_File = open('Positive_Tweets1.txt','r')
for line in positiveTweets_File:
    positiveTweets.append(json.loads(line.strip('\n')))
print "Total positive records are " + str(len(positiveTweets))

#Collect negative tweets
negativeTweets = []
negativeTweets_File = open('Negative_Tweets1.txt', 'r')
for line in negativeTweets_File:
    negativeTweets.append(json.loads(line.strip('\n')))
print "Total negative records are " + str(len(negativeTweets))

#Convert tweets to data frame to extract required information
#DataFrame for positive tweets
ptweets = pd.DataFrame()
ptweets['text'] = map(lambda tweet: tweet['text'], positiveTweets)

#DataFrame for negative tweets
ntweets = pd.DataFrame()
ntweets['text'] = map(lambda tweet: tweet['text'], negativeTweets)

# Extract the vocabulary of keywords from positive tweets
positiveVocab = dict()
for text in ptweets['text']:
    for term in text.split():
        term = term.lower()
        if len(term) > 2 and term not in stopwords:
            if positiveVocab.has_key(term):
                positiveVocab[term] = positiveVocab[term] + 1
            else:
                positiveVocab[term] = 1

# Remove terms whose frequencies are less than a threshold (e.g., 20)
positiveVocab = {term: freq for term, freq in positiveVocab.items() if freq > 20}
# Generate an id (starting from 0) for each term in vocab
positiveVocab = {term: idx for idx, (term, freq) in enumerate(positiveVocab.items())}
print "Length of positive vocab : " + str(len(positiveVocab))
print positiveVocab

# Extract the vocabulary of keywords from negative tweets
negativeVocab = dict()
for text in ntweets['text']:
    for term in text.split():
        term = term.lower()
        if len(term) > 2 and term not in stopwords:
            if negativeVocab.has_key(term):
                negativeVocab[term] = negativeVocab[term] + 1
            else:
                negativeVocab[term] = 1

# Remove terms whose frequencies are less than a threshold (e.g., 35)
negativeVocab = {term: freq for term, freq in negativeVocab.items() if freq > 20}
# Generate an id (starting from 0) for each term in vocab
negativeVocab = {term: idx for idx, (term, freq) in enumerate(negativeVocab.items())}
print "Length of negative vocab : " + str(len(negativeVocab))
print negativeVocab

#combine both vocab
vocab = dict()
vocab = dict(chain.from_iterable(vocab.iteritems() for vocab in(positiveVocab, negativeVocab)))
vocab = {term: idx for idx, (term, freq) in enumerate(vocab.items())} #Generate id for each term in vocab
print "Length of all vocab : " + str(len(vocab))

# Generate X and y
X = []
Y = []
for text in ptweets['text']:
    x = [0] * len(vocab)
    terms = [term for term in text]
    for term in terms:
        if vocab.has_key(term):
            x[vocab[term]] += 1
    Y.append(1)
    X.append(x)

for text in ntweets['text']:
    x = [0] * len(vocab)
    terms = [term for term in text]
    for term in terms:
        if vocab.has_key(term):
            x[vocab[term]] += 1
    Y.append(0)
    X.append(x)

# Folder cross validation
clf = LogisticRegression()
clf.fit(X, Y)

# Collect testing tweets
tweets = []
for line in open('Testing_Tweets1.txt').readlines():
    tweets.append(json.loads(line.strip('\n')))

# DataFrame for testing tweets
ttweets = pd.DataFrame()
ttweets['date'] = map(lambda tweet: tweet['created_at'], tweets)
ttweets['text'] = map(lambda tweet: tweet['text'], tweets)

# Generate X for testing tweets
X = []
for text in ttweets['text']:
    x = [0] * len(vocab)
    terms = [term for term in text.split() if len(term) > 2]
    for term in terms:
        if vocab.has_key(term):
            x[vocab[term]] += 1
    X.append(x)
y = clf.predict(X)

f = open('LR_predictions.txt', 'wb')
idx = 0
for date in ttweets['date']:
    f.write(json.dumps([date, y[idx]]) + '\r\n')
    idx += 1
f.close()

print '\r\nAmong the total {1} tweets, {0} tweets are predicted as positive.'.format(sum(y), len(y))
