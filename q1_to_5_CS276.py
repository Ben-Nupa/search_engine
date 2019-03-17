# This script loads the data from CS276 to count the tokens of the dataset

import os
import pickle
from math import log
import matplotlib.pyplot as plt
from index import *
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


def fillDict(directory, tokens):
    """Fill the dictionnay 'tokens' with tokens in the specified directory without any token in the list common_words"""
    listFiles = os.listdir(directory)
    for fileName in listFiles:
        file = open(os.path.join(directory, fileName), "r")
        content = file.readlines()
        for line in content:
            words = Index.normalize(line)
            for word in words:
                tokens[word] = tokens.get(word, 0) + 1

def analysis(dir_path, redo_tokenization=False):
    tokens = dict()
    
    if redo_tokenization:
        print("Reading the dataset")
        for i in range(5):
            directory = os.path.join(dir_path, str(i))
            print("Reading files from directory " + str(i) + " ...")
            fillDict(directory, tokens)
        save_dictionnary("dictionnaireHalf.pkl",tokens)
        for i in range(5,10):
            directory = os.path.join(dir_path, str(i))
            print("Reading files from directory " + str(i) + " ...")
            fillDict(directory, tokens)
        save_dictionnary(os.path.join(dir_path, "dictionnaire.pkl"), tokens)
    else :
        print("Checking the saved results without reading again the dataset")
    
    tokens = load_saved_dictionnary(os.path.join(dir_path, "dictionnaire.pkl"))
    
    #Question1
    print("Question 1 : "+str(sum(tokens.values()))+" terms in the vocabulary")
    
    #Question2
    print("Question 2 : "+str(len(tokens))+" distinct tokens")
    
    #Question3
    T = sum(tokens.values())
    V = len(tokens)
    tokens_half = load_saved_dictionnary("dictionnaireHalf.pkl")
    Tp = sum(tokens_half.values())
    Vp = len(tokens_half)
    b = (log(V) - log(Vp)) / (log(T) - log(Tp))
    k = V/(T**b)
    print("Question 3 : Heaps Law : k = {} and b = {}".format(k,b))
    
    #Question4
    print("Question 4 : For 1 million tokens, the vocabulary size would be {}".format(int(k*1e6**b)))
    
    #Question5
    print("Question 5")
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

