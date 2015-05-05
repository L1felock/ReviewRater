from nltk.tag import pos_tag
import json

CD = "cardinal digit"
DT = "determiner"
JJ = "adjective"
JJS = "superlative adj.; biggest, etc."
JJR = "comparative adjectives"
IN = "preposition, subordinating conjunction"
LS = "list marker"
MD = "modal; could, will, etc."
NN = "singular noun"
NNS = "plural noun"
PRP = "personal pronoun	I, he, she"
PRP_cash = "possessive pronoun"
RB = "adverb"
VB = "verb"
VBD = "past tense verb"
VBG = "verb, gerund/present participle	taking"
VBN = "verb, past participle"
VBP = "verb, present tense"
VBZ = "verb; 3rd person singular present"
WP = "wh-pronoun	who, what"

# classifies all the terms in a given file
# writes classification to a file
def classify_terms(filename):
    # open file for reading
    file = open(filename, 'r')
    # load all info into a dictionary
    term_dict = json.load(file)
    # close file
    file.close()
    # open file for writing
    output = open("term_classes.txt", 'w')
    # step through dictionary, classifying each term
    for key in term_dict:
        # puts key in a list so that the pos_tag function will classify it properly
        list = [key]
        # classify token according to its purpose
        token_classification = pos_tag(list)
        # access results
        token = token_classification[0][0]
        classification = token_classification[0][1]
        # create results string
        results = token + " " + classification + "\n"
        # write results to file
        output.write(results)

    output.close()

def find_ratios(filename):
    # open data file for processing
    data = open(filename, 'r')
    # step through data line by line
    for line in data:
        # strip whitespace
        line = line.strip()
        # tokenize string into a list
        line = line.split()
        # increment count
        if line[1] in count_dict:
            count_dict[line[1]] += 1
        else:
            count_dict[line[1]] = 1

# file which contains the terms to be classified
filename = "unstem_freq.txt"
#filename = "termfreq.txt"
term_input = 'term_classes.txt'
count_dict = {}
find_ratios(term_input)
print(count_dict)