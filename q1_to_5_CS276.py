# This script loads the data from CS276 to count the tokens of the dataset

import os
import pickle
from math import log
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

#======================================FIRST PART : TOKENIZATION================================================================

tokenize = False  # Set to True if you want to do again the tokenization and the dictionnary, otherwise, the saved dictionnary will be loaded


def load_saved_dictionnary(file_name):
    """Load the saved dictionnary with pickle"""
    with open(file_name, 'rb') as handle:
        dict = pickle.load(handle)
    return dict


def save_dictionnary(file_name, dict_object):
    """Save the dictionnary"""
    with open(file_name, 'wb') as handle:
        pickle.dump(dict_object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_common_words(filePath):
    """Returns the list of the common words listed in the specified file path"""
    file = open(filePath, "r")
    return file.read().splitlines()


def fillDict(directory, tokens, common_words):
    """Fill the dictionnay 'tokens' with tokens in the specified directory without any token in the list common_words"""
    listFiles = os.listdir(directory)
    for fileName in listFiles:
        file = open(os.path.join(directory, fileName), "r")
        content = file.readlines()
        for line in content:
            words = line.split(" ")
            for word in words:
                if word not in common_words:
                    tokens[word] = tokens.get(word, 0) + 1


tokens = dict()
common_words = read_common_words(os.path.join("..", "common_words"))
if tokenize:
    print("Reading the dataset")
    for i in range(5):
        directory = os.path.join("..", "pa1-data", "pa1-data", str(i))
        print("Reading files from directory " + str(i) + " ...")
        fillDict(directory, tokens, common_words)
    save_dictionnary("dictionnaireHalf.pkl",tokens)
    for i in range(5,10):
        directory = os.path.join("..", "pa1-data", "pa1-data", str(i))
        print("Reading files from directory " + str(i) + " ...")
        fillDict(directory, tokens, common_words)
    save_dictionnary("dictionnaire.pkl", tokens)
else :
    print("Checking the saved results without reading again the dataset")


tokens = load_saved_dictionnary("dictionnaire.pkl")

#Question1
print("Question 1 : "+str(sum(tokens.values()))+" terms in the vocabulary")

#Question2
print("Question 2 : "+str(len(tokens))+" distinct tokens")

#Question3
M_tot = sum(tokens.values())
T_tot = len(tokens)
tokens_half = load_saved_dictionnary("dictionnaireHalf.pkl")
M_half = sum(tokens_half.values())
T_half = len(tokens_half)
b = (log(M_tot) - log(M_half)) / (log(T_tot) - log(T_half))
k = M_tot/(T_tot**b)
print("Question 3 : Heaps Law : k = {} and b = {}".format(k,b))

#Question4
print("Question 4 : For 1 million tokens, the vocabulary size would be {}".format(int(k*1e6**b)))

#Question5
frequencies = sorted(tokens.values(),reverse=True)
ranks = [i+1 for i in range(len(frequencies))]
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(ranks,frequencies, color='blue', lw=2)
ax1.set_xlabel("Rang")
ax1.set_ylabel("Fréquence")
ax1.set_title("Graphe fréquence vs rang")
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(ranks,frequencies, color='blue', lw=2)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel("Rang")
ax2.set_ylabel("Fréquence")
ax2.set_title("Graphe fréquence vs rang (echelle log)")
plt.show()

#======================================SECOND PART : INDEX================================================================

def create_term_ids(tokens) :
    """Return a dictionary term_id/term using the dictionary of tokens"""
    terms = tokens.copy()
    term_ids = dict()
    i = 0
    for word,count in terms.items() :
        term_ids[word] = i
        i = i+1
    return term_ids

def create_doc_ids(directory) :
    """Return a dictionary doc_ids/doc in the specified directory"""
    list_files = os.listdir(directory)
    doc_ids = dict()
    i = 0
    for doc in list_files:
        doc_ids[doc] = i
        i = i+1
    return doc_ids

def create_term_doc_matrix(directory,term_ids,doc_ids) :
    """Return a sparse matrix containing pair of term_ids and doc_ids"""
    terms = []
    docs = []
    count = []
    listFiles = os.listdir(directory)
    for fileName in listFiles:
        file = open(os.path.join(directory, fileName), "r")
        tokens_in_file = dict() #Temporary dictionnary to count the frequency of each token in the document
        content = file.readlines()
        for line in content:
            words = line.split(" ")
            for word in words:
                if word not in common_words:
                    tokens_in_file[word] = tokens_in_file.get(word, 0) + 1
                    terms.append(term_ids[word])
                    docs.append(doc_ids[fileName])
                    count.append(tokens_in_file[word])
    terms = np.array(terms)
    docs = np.array(docs)
    count = np.array(count)
    matrix_term_doc = sparse.coo_matrix((count,(terms,docs)))
    return matrix_term_doc


term_ids = create_term_ids(tokens)
directory = os.path.join("..", "pa1-data", "pa1-data", str(1))
doc_ids = create_doc_ids(directory)
#matrix = create_term_doc_matrix(directory,term_ids,doc_ids)