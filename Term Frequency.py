from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import json

freqDict = {}

# tokenize a string, returns a list
def tokenizeDocument(string):
    myTokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokens = myTokenizer.tokenize(string)
    return tokens

# removes stopwords from a list of tokens
def removeStopWords(someTokens):
    myStopWords = stopwords.words('english')

    finalTokens = []
    for token in someTokens:
        if token not in myStopWords:
            finalTokens.append(token)

    return finalTokens

# stems a list of tokens using Porter stemmer
def stemTokens(givenTokens):
    myStemmer = PorterStemmer()
    i = 0
    while(i < len(givenTokens)):
        givenTokens[i] = myStemmer.stem(givenTokens[i])
        i = i + 1
    return givenTokens

# counts the instances of a term, stores in dictionary
def getTermFrequency(tokenList):


    for token in tokenList:
        if token in freqDict:
            freqDict[token] += 1
        else:
            freqDict[token] = 1


# read all files
file = open('testSet.json', 'r')
for line in file:
    line = json.loads(line)
    line = line['text'].lower()
    line = line.strip()
    list = tokenizeDocument(line)
    tokens = removeStopWords(list)
    getTermFrequency(tokens)
file = open('trainSet.json', 'r')
for line in file:
    line = json.loads(line)
    line = line['text'].lower()
    line = line.strip()
    list = tokenizeDocument(line)
    tokens = removeStopWords(list)
    getTermFrequency(tokens)

# store term frequency dictionary in file
json.dump(freqDict, open('unstem_freq.txt', 'w'))


'''
limit = 1000
file = open("unstem_freq.txt", 'r')
final_dict = {}
freqDict = json.load(file)
for key in freqDict:
    if freqDict[key] > limit:
        final_dict[key] = freqDict[key]

order = sorted(final_dict.items(), key=lambda x: x[1])

print(order)
print(len(order))'''