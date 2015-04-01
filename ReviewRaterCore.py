import json
import time
import os


def divyData(openFile, testFile, trainFile, corpusRoot):
#parses the datafile from the yelp dataset challenge
    openTestFile = open(os.path.join(corpusRoot, testFile), "w")
    openTrainFile = open(os.path.join(corpusRoot, trainFile), "w")

    line = ""
    line = openFile.readline()
    usefulLineCount = 0
    while(line): #breaks when the line is empty
        #use ['votes']['useful']  and   ['text'] to reference the number of useful
        #votes and the text of the review
        lineDict = json.loads(line)
        if lineDict['votes']['useful'] > 10:
            if usefulLineCount % 2 == 0:
                openTestFile.write(line)
                usefulLineCount = usefulLineCount + 1
            else:
                openTrainFile.write(line)
                usefulLineCount = usefulLineCount + 1
        line = openFile.readline()
        
        
    openTrainFile.close()
    openTestFile.close()
    #doesn't return anything


def main():
    corpusRoot = 'C:\\Users\\Joey\\Desktop\\Yelp Dataset'   ######MODIFY ME#####
    dataFile = "yelp_academic_dataset_review.json"
    testFile = "testSet.json"
    trainFile = "trainSet.json"    
    response = ""
    start = time.time() #used to calculate runtime
    
    openFile = open(os.path.join(corpusRoot, dataFile), "r")
    while(1):    
        response = raw_input("do we need to divy up the corpus? y/n: ")
        if response == "y":
            divyData(openFile, testFile, trainFile, corpusRoot)
        elif response == "n":
            break
        else:
            print("please give valid input")
    openFile.close()    
    
        
    stop = time.time()
    runTime = stop - start #measured in seconds
    print("the runtime was: " + runTime + " seconds")
    


if __name__ == "__main__":
    main()