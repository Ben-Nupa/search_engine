#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from math import log
import numpy as np
import matplotlib.pyplot as plt
from time import time
    
    
    
def text_preprocessing(text):
    text.replace(" ", " ")
    text.replace(";", " ")
    text.replace(",", " ")
    text.replace(".", " ")
    text.replace("-", " ")
    text.replace("_", " ")
    text.replace("\"", " ")
    text.replace("\'", " ")
    text.replace("!", " ")
    text.replace("?", " ")
    text.replace("[", " ")
    text.replace("]", " ")
    text.replace("(", " ")
    text.replace(")", " ")
    text.replace("#", " ")
    text.lower()
    return text.split(" ")
    
    

def BSBI():
    """
    doit retourner :
    un dictionnaire (token:token_id)
    un dictionnaire (token_id:[liste de (doc_id, nb_occ)])
    optionnel : un dictionnaire (doc_id:[liste de booleen qui indique si le terme i est présent])
    """
    raw = np.loadtxt("cacm.all", dtype=str, delimiter="someneverhappeningstr")
    keep = False
    
    max_tok_id = 0
    tok2id = {}
    doc_id = 0
    tokdoc = {}
    
    V = 0  # nb of distinct words in total data
    T = 0  # nb of tokens in total data
    
    for row in raw:
        if row.startswith(".") :
            if row.startswith(".I"):
                doc_id += 1 # assumption : docs are read in order
                
            keep = row.startswith(".T") or row.startswith(".W") or row.startswith(".K")
                
        elif keep:
            terms = text_preprocessing(row)
            
            for term in terms :
                if term != "":
                    if term not in tok2id:
                        tok2id[term] = max_tok_id
                        max_tok_id += 1
                        tokdoc[term] = [0 for i in range(doc_id)] # the token was never seen before in the preceeding docs
                    if term not in term_dic:
                        term_dic[term] = 0
                    term_dic[term] += 1
    
    
    
    
def analysis():
    raw = np.loadtxt("cacm.all", dtype=str, delimiter="someneverhappeningstr")
    keep = False
    
    term_dic = {}
    doc_id = 0
    nb_docs = 3204  # nb of docs in cacm.all
    
    V = 0  # nb of distinct words in total data
    T = 0  # nb of tokens in total data
    
    Vp = 0  # nb of distinct words in half data
    Tp = 0  # nb of tokens in half data
    
    # Heaps law parameters : V = k * T**b
    k = 0
    b = 0
    
    for row in raw:
        if row.startswith(".") :
            if row.startswith(".I"):
                doc_id += 1 # assumption : docs are read in order
                if doc_id >= nb_docs/2 and doc_id < nb_docs/2 + 1:
                    print("At half the docs : ")
                    print("Vocabulary size : {}".format(len(term_dic)))
                    print("Nb tokens : {}".format(sum(term_dic.values())))
                    Vp = len(term_dic)
                    Tp = sum(term_dic.values())
            if row.startswith(".T") or row.startswith(".W") or row.startswith(".K"):
                keep = True
            else :
                keep = False
        elif keep:
            terms = text_preprocessing(row)
            
            for term in terms :
                if term != "":
                    if term not in term_dic:
                        term_dic[term] = 0
                    term_dic[term] += 1
        
    print("Total")
    print("Vocabulary size : {}".format(len(term_dic)))
    print("Nb tokens : {}".format(sum(term_dic.values())))
    V = len(term_dic)
    T = sum(term_dic.values())
    
    b = (log(V) - log(Vp)) / (log(T) - log(Tp))
    k = V/(T**b)
    print("Heaps parameters :")
    print("k : {}".format(k))
    print("b : {}".format(b))
    print("checking the heaps parameters (the 2 following numbers should be equal):")
    print(k * (T**b))
    print(V)
    
    print("If the data had 1M tokens, it would have roughly {} distinct words".format(int(k*1e6**b)))
    
    s = time()
    print("Sorting the frequencies...")
    term_list = sorted(term_dic.items(), key=lambda x: x[1], reverse=True)
    print("Done ! {:.2f}s".format(time()-s))
    
    print(term_list[0])
    print(term_list[-1])
    words = [w for w,f in term_list]
    freq = [f for w,f in term_list]
    rank = [i for i in range(len(freq))]
    logrank = [log(i+1) for i in range(len(freq))]
    logfreq = [log(f+1) for w,f in term_list]

    plt.figure(1)
    plt.subplot(211)
    plt.plot(rank, freq)
    plt.ylabel("word frequency")
    plt.xlabel("rank")

    plt.subplot(212)
    plt.plot(logrank, logfreq)
    plt.ylabel("log word frequency")
    plt.xlabel("log rank")
    plt.show()
    
    
if __name__ == "__main__":
    analysis()
    BSBI()