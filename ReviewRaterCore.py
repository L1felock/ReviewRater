import json
import time
import os
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import math
#make all the reviews into TFIDF vectors and find cosine similarity between the
#review to classify and all the other reviews. take the majority vote on 
#classification


#CONSTANTS-----------------------------------
USEFULTHRESHOLD = 3 #represents the number of upvotes a review has to get before it
#is considered useful
CORPUSROOT = 'C:\\Users\\Joey\\Desktop'
TESTFILE = "testSet.json"
TRAINFILE = "trainSet.json"
CORPUSFILE = "yelp_academic_dataset_review.json"


def divyData(openFile, testFile, trainFile, corpusRoot):
#parses the datafile from the yelp dataset challenge and makes a test and
#training set file
    openTestFile = open(os.path.join(corpusRoot, testFile), "w")
    openTrainFile = open(os.path.join(corpusRoot, trainFile), "w")
    
    totalLineCount = 0
    line = ""
    line = openFile.readline()
    usefulLineCount = 0
    while(line): #breaks when the line is empty
        #use ['votes']['useful']  and   ['text'] to reference the number of useful
        #votes and the text of the review
        lineDict = json.loads(line)
        if lineDict['votes']['useful'] > USEFULTHRESHOLD:
            if usefulLineCount % 2 == 0:
                openTestFile.write(line)
                usefulLineCount = usefulLineCount + 1
            else:
                openTrainFile.write(line)
                usefulLineCount = usefulLineCount + 1
        line = openFile.readline()
        totalLineCount = totalLineCount + 1
        
    print("the number of useful lines was: " + str(usefulLineCount))  
    print("the total number of lines was: " + str(totalLineCount))
    openTrainFile.close()
    openTestFile.close()
    #doesn't return anything
    
def tokenizeDocument(string):
    myTokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokens = myTokenizer.tokenize(string)
    return tokens

def removeStopWords(someTokens):
    myStopWords = stopwords.words('english')
    
    finalTokens = []
    for token in someTokens:
        if token not in myStopWords:
            finalTokens.append(token)
    
    return finalTokens
    
def stemTokens(givenTokens):
    myStemmer = PorterStemmer()
    i = 0
    while(i < len(givenTokens)):
        givenTokens[i] = myStemmer.stem(givenTokens[i])
        i = i + 1
    return givenTokens
    
def stemToken(givenToken):
    stemmer = PorterStemmer()
    return stemmer.stem(givenToken)
    
def getIDF(term, reviewsVector):
#takes a dictionary of review vectors and gets the IDF of a term with respect
#to that dictionary of review vectors
    term = term.lower()
    term = stemToken(term)

    documentFrequency = 0
    for reviewIndex in reviewsVector:
        if(term in reviewsVector[reviewIndex].keys()):
            documentFrequency = documentFrequency + 1
    if(documentFrequency == 0):
        return 0
    else:
        return math.log10(((len(reviewsVector))/documentFrequency))
        
def normalizeVector(vector):
#normalizes a vector to unit length
    vectorLength = 0
    normalVector = dict()
    for term in vector:
        vectorLength = vectorLength + (vector[term] * vector[term])
    vectorLength = math.sqrt(vectorLength)
    for term in vector:
        if vectorLength > 0:
            normalVector[term] = vector[term]/vectorLength
        else:
            print("there is probably an issue with parsing")
            normalVector[term] = 0 #pretty sure this is supposed to be zero
    return normalVector
        
def docDocSim(vec1, vec2):
#return the cosine similarity betwen two review vectors
    similarity = 0
    for term in vec1:
        if term in vec2:
            similarity = similarity + (vec1[term] * vec2[term]) #compiling the numerator
            
    return similarity

def convertToTFIDF(vec): 
#takes a vector of term frequencies of a review and weights them
    #tempIndex = normalizeVector(tempIndex)
    returnVector = dict()
    for term in vec:
        returnVector[term] = ((1 + math.log10(vec[term])) * getIDF(term))
    return normalizeVector(returnVector)

def getTermFrequency(tokenList):
    freqDict = {}

    for token in tokenList:
        if token in freqDict:
            freqDict[token] += 1
        else:
            freqDict[token] = 1

    return freqDict

def processTrainingReviews(trainingReviews):
        openFile = open(os.path.join(CORPUSROOT, TRAINFILE), "r")
        tokenDict = {}
        reviewVec = {}

        i = 0
        for line in openFile:
            dictLine = json.loads(line)
            review = dictLine["text"]

            reviewList = tokenizeDocument(review)
            reviewTokens = removeStopWords(reviewList)
            stemReview = stemTokens(reviewTokens)

            tokenDict = getTermFrequency(stemReview)
            reviewVec = convertToTFIDF(tokenDict)

            trainingReviews[i] = reviewVec
            i += 1



def main(): 
    response = ""
    start = time.time() #used to calculate runtime
    
    openFile = open(os.path.join(CORPUSROOT, CORPUSFILE), "r")
    while(1):    
        response = input("do we need to divy up the corpus? y/n: ")
        if response == "y":
            divyData(openFile, TESTFILE, TRAINFILE, CORPUSROOT)
            break
        elif response == "n":
            break
        else:
            print("please give valid input")
    openFile.close()

    trainingReviews = {}
    processTrainingReviews(trainingReviews)
    
        
    stop = time.time()
    runTime = stop - start #measured in seconds
    print("the runtime was: " + str(runTime) + " seconds")
    


if __name__ == "__main__":
    main()