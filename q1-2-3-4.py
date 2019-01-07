#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from math import log
import numpy as np
import matplotlib.pyplot as plt
import os
    
if __name__ == "__main__":
    filename = os.path.join('data', 'CACM', 'cacm.all')
    
    raw = np.loadtxt(filename, dtype=str, delimiter="someneverhappeningstr")
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
            row = row.lower()
            row.replace("; ", " ")
            row.replace(", ", " ")
            row.replace("(", "")
            row.replace(")", "")
            
            terms = row.split(" ")  # TODO must split along other word sep too
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
    
    term_list = sorted(term_dic.items(), key=lambda x: x[1], reverse=True)
    print(term_list[0])
    print(term_list[-1])
    
    
    
    
    
    
    
    