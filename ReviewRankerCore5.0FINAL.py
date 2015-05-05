import json
import time
import os
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.tag import pos_tag
import math

#make all the reviews into TFIDF vectors and find cosine similarity between the
#review to rank and all the other reviews. take the majority vote on 
#classification


class Review:
    def __init__(self, review, score, actualRating):
        self.review = review
        self.score = score
        self.actualRating = actualRating
        

#CONSTANTS----------------------------------- ONLY CONSTANTS WITH COMMENTS AFTER THEM SHOULD BE CHANGED
USEFULTHRESHOLD = 20 #represents the number of upvotes a review has to get before it
#is considered useful -- divy the corpus if changed
CORPUSROOT = 'C:\\Users\\Joey\\Desktop\\DataMining Project Results\\' #change me! -- divy corpus if changed
TESTFILE = "testSet.json"
TRAINFILE = "trainSet.json"
CORPUSFILE = "yelp_academic_dataset_review.json"
PROCESSEDTRAINFILE = "objectStorage.json"
METRICCOMPILATION = "metricComp.txt"
OUTFILE = "results.txt"
TESTCOUNT = 1000 #number of reviews to be ranked --- divy corpus if changed
KCOUNT = 20 #number of k nearest neighbors CURRENTLY UNUSED

def removeNonAscii(s): 
#removes non-ascii characters from a string
    return "".join(i for i in s if ord(i)<128)

def divyData(openFile, testFile, trainFile, corpusRoot):
#parses the datafile from the yelp dataset challenge and makes a test and
#training set raw file
    openTestFile = open(os.path.join(corpusRoot, testFile), "w")
    openTrainFile = open(os.path.join(corpusRoot, trainFile), "w")
    
    testLineCount = 0
    totalLineCount = 1
    line = ""
    line = (openFile.readline())
    usefulLineCount = 0
    while(line): #breaks when the line is empty
        #use ['votes']['useful']  and   ['text'] to reference the number of useful
        #votes and the text of the review
        lineDict = json.loads(line)
        if lineDict['votes']['useful'] > USEFULTHRESHOLD:
            if usefulLineCount % 2 == 0:
                openTestFile.write(removeNonAscii(line))
                usefulLineCount = usefulLineCount + 1
            else:
                openTrainFile.write(removeNonAscii(line))
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
    
    i = 0
    while(i < len(tokens)):
        tokens[i] = tokens[i].lower()
        i = i + 1
        
    return tokens

def removeStopWords(someTokens):
#also removes nouns in this version
    myStopWords = stopwords.words('english')
    
    finalTokens = []
    for token in someTokens:
        if token not in myStopWords:
            finalTokens.append(token)
            
    #removing the nouns from the corpus
    #pos_tag returns a dict where the first key (0) corresponds to the word
    #and the second key (1) corresponds to what kind of word it is (noun etc)
    #documentation for this function here: http://www.nltk.org/book/ch05.html
    """
    finalFinalTokens = []
    nounClassifications = ["NN", "NNS", "PRP"] #NN = singular noun
    markedTokens = pos_tag(finalTokens)
    for pair in markedTokens:
        if pair[1] not in nounClassifications:
            print(pair[0])
            finalFinalTokens.append(pair[0])
    """
    
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
#MAKE SURE THE TERMS ARE STEMMED BEFORE THEY GET HERE
    documentFrequency = 0
    for reviewIndex in reviewsVector:
        if(term in reviewsVector[reviewIndex].keys()):
            documentFrequency = documentFrequency + 1
    if(documentFrequency == 0):
        #print("idf returned 0")
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
            normalVector[term] = 0 #pretty sure this is supposed to be zero
    return normalVector
        
def docDocSim(vec1, vec2):
#return the cosine similarity betwen two review vectors
    similarity = 0
    for term in vec1:
        if term in vec2:
            similarity = similarity + (vec1[term] * vec2[term])
            
    return similarity

def convertToTFIDF(vec): 
#takes a vector of reviews and weights the terms in them
    #tempIndex = normalizeVector(tempIndex)
    returnVector = vec
    
    for reviewIndex in vec:
        for termIndex in vec[reviewIndex]:
            returnVector[reviewIndex][termIndex] = ((1 + math.log10(vec[reviewIndex][termIndex])) * getIDF(termIndex, vec))
       
    for reviewIndex in returnVector:
        returnVector[reviewIndex] = normalizeVector(returnVector[reviewIndex])
        
    return returnVector
    
def convertReviewToTFIDF(review, vec):
#converts a review to TFIDF using the corpus (vec) to generate IDF

    for termIndex in review:
        review[termIndex] = ((1 + math.log10(review[termIndex])) * getIDF(termIndex, vec))
    
    return normalizeVector(review)
    
def score(reviewVec, trainingReviews):
#takes a normalized review TFIDF vector and gives it a score based on the summation of the
#cosine similarities between itself and known positive reviews in the training set
#reviewVec is the review itself (weighted and normalized) while trainingreviews is a dict of reviews
    currentDistance = 0
    mySum = 0
    for index in trainingReviews:
        currentDistance = docDocSim(trainingReviews[index], reviewVec)
        mySum += currentDistance
    
    return (mySum)
    

def getTermFrequency(tokenList):
    freqDict = {}

    for token in tokenList:
        if token in freqDict:
            freqDict[token] += 1
        else:
            freqDict[token] = 1

    return freqDict



#returns the training review vector
"""
validated and working correctly
"""
def processReviews():
    trainingReviews = dict()
    trainingReviewsTF = dict()
    openFile = open(os.path.join(CORPUSROOT, TRAINFILE), "r")
    i = 0
    for line in openFile:
        dictLine = json.loads(line)
        review = dictLine["text"]

        reviewList = tokenizeDocument(review)
        reviewTokens = removeStopWords(reviewList)
        stemReview = stemTokens(reviewTokens)

        trainingReviewsTF[i] = getTermFrequency(stemReview)
        i += 1


    trainingReviews = convertToTFIDF(trainingReviewsTF) 
    openTrainFile = open(os.path.join(CORPUSROOT, PROCESSEDTRAINFILE), "w") 
    openTrainFile.write(json.dumps(trainingReviews))
    openTrainFile.close()          
        
    return trainingReviews


def main(): 
    response = ""
    trainingReviews = dict()
    start = time.time() #used to calculate runtime
    rankedReviewList = []
    
    openFile = open(os.path.join(CORPUSROOT, CORPUSFILE), "r")
    while(1):    
        response = "y"
        #response = input("do we need to divy up the corpus? y/n: ")
        if response == "y":
            print("divying the data")
            divyData(openFile, TESTFILE, TRAINFILE, CORPUSROOT)
            trainingReviews = processReviews()
            break
        elif response == "n":
            openFile = open(os.path.join(CORPUSROOT, PROCESSEDTRAINFILE), "r")
            trainingReviews = json.loads(openFile.readline())
            openFile.close()
            break
        else:
            print("please give valid input")
    openFile.close()

    
    #getting and ranking the reviews in the test set
    print("ranking the test set...")
    openFile = open(os.path.join(CORPUSROOT, TESTFILE), "r")
    test = {}
    for line in openFile:
        test = json.loads(line)
        reviewText = test["text"]
        reviewRating = test['votes']['useful']
        reviewTokens = tokenizeDocument(test["text"])
        reviewTokens = removeStopWords(reviewTokens)
        reviewTokens = stemTokens(reviewTokens)
        
        review = getTermFrequency(reviewTokens)
        review = convertReviewToTFIDF(review, trainingReviews)
        rankedReviewList.append(Review(reviewText, score(review, trainingReviews), reviewRating))
    
    #sorting the reviews by their cosine similarity scores
    rankedReviewList.sort(key=lambda x: x.score, reverse=False)
    revLength = len(rankedReviewList)    
    
    
    
    #these are rough calculations of estimates of the metrics
    q1OrderedUsefulness = rankedReviewList[0:round(.25*revLength)]
    q1OrderedUsefulness.sort(key=lambda x: x.actualRating, reverse=False)
    q2OrderedUsefulness = rankedReviewList[round(.25*revLength):round(.5*revLength)]
    q2OrderedUsefulness.sort(key=lambda x: x.actualRating, reverse=False)
    q3OrderedUsefulness = rankedReviewList[round(.5*revLength):round(.75*revLength)]
    q3OrderedUsefulness.sort(key=lambda x: x.actualRating, reverse=False)
    q4OrderedUsefulness = rankedReviewList[round(.75*revLength):(revLength-1)]
    q4OrderedUsefulness.sort(key=lambda x: x.actualRating, reverse=False)




    #getting quarterly usefulness summations to estimate usefulness of the ranker 
    outfile = open(os.path.join(CORPUSROOT, OUTFILE), "w")
    metrics = open(os.path.join(CORPUSROOT, METRICCOMPILATION), "a")
    metrics.write(OUTFILE + " results: \n\n")
    i = 0
    usefulVotes = 0
    revLength = len(rankedReviewList)
    while i < (.25*revLength):
        usefulVotes += rankedReviewList[i].actualRating                
        i = i + 1
    outfile.write("number of useful votes in each quarter:\n")
    outfile.write("first quarter: " + str(usefulVotes) + "\n")
    outfile.write("first quarter median: " + str(rankedReviewList[round(.125*revLength)].actualRating) + "\n")
    metrics.write("first quarter: " + str(usefulVotes) + "\n")
    metrics.write("first quarter median: " + str(q1OrderedUsefulness[round(.5*len(q1OrderedUsefulness))].actualRating) + "\n")    
    
    usefulVotes = 0
    while i < (.5*revLength):
        usefulVotes += rankedReviewList[i].actualRating
        i = i + 1
    outfile.write("second quarter: " + str(usefulVotes) + "\n")
    outfile.write("second quarter median: " + str(rankedReviewList[round(.375*revLength)].actualRating) + "\n")
    metrics.write("second quarter: " + str(usefulVotes) + "\n")
    metrics.write("second quarter median: " + str(q2OrderedUsefulness[round(.5*len(q2OrderedUsefulness))].actualRating) + "\n")
        
    usefulVotes = 0
    while i < (.75*revLength):
        usefulVotes += rankedReviewList[i].actualRating
        i = i + 1
    outfile.write("third quarter: " + str(usefulVotes) + "\n")
    outfile.write("third quarter median: " + str(rankedReviewList[round(.625*revLength)].actualRating) + "\n")
    metrics.write("third quarter: " + str(usefulVotes) + "\n")
    metrics.write("third quarter median: " + str(q3OrderedUsefulness[round(.5*len(q3OrderedUsefulness))].actualRating) + "\n")
    
    usefulVotes = 0
    while i < (revLength):
        usefulVotes += rankedReviewList[i].actualRating
        i = i + 1
    outfile.write("fourth quarter: " + str(usefulVotes) + "\n")
    outfile.write("fourth quarter median: " + str(rankedReviewList[round(.875*revLength)].actualRating) + "\n")
    metrics.write("fourth quarter: " + str(usefulVotes) + "\n")
    metrics.write("fourth quarter median: " + str(q4OrderedUsefulness[round(.5*len(q4OrderedUsefulness))].actualRating) + "\n")
    metrics.write("===============================end metrics==================================\n\n")
    
    metrics.write("====================================================\n\n")
    metrics.close()
    





    #printing out individual review results
    for review in rankedReviewList:
        outfile.write(removeNonAscii(review.review))
        outfile.write("\n")
        outfile.write("-------------------\n")
        outfile.write("similarity summation: " + str(review.score) + "\n")
        outfile.write("actual Rating: " + str(review.actualRating) + "\n")
        outfile.write("===================\n\n\n")

    outfile.close()
    
    
        
    stop = time.time()
    runTime = stop - start #measured in seconds
    print("the runtime was: " + str(runTime) + " seconds")
    
    

    


if __name__ == "__main__":
    print("make sure that you have a file named \"metricComp.txt\" in your corpus root and clear it if you want fresh data")
    response = ""
    while 1:
        response = input("would you like to cycle the variables for data metric collection? (y/n): ")
        
        if response == "y":
            i = 50
            while(i >= 10):
                OUTFILE = "resultsThresh" + str(i) + ".txt"
                USEFULTHRESHOLD = i
                main()
                i -= 1
            break
        elif response == "n":
            response = input("what would you like the number of useful votes to be considered useful to be?(positive integer please): ")
            USEFULTHRESHOLD = int(response)
            main()
            break
        else:
            print("please give a valid response")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    