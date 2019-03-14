import numpy as np
import re
class Node():
    """
    Node class for a tree representing a boolean query.
    Each node is an operation, each leaf is a token_id
    """
    
    def __init__(self, keyword, left=None, right=None):
        self.keyword = keyword
        
        if keyword in ["not", "and", "or"]:
            
            if right is None:
                raise ArgumentError("no right subtree for this operator")
            self.right = right
            if keyword == "or" or keyword == "and":
                if left is None:
                    raise ArgumentError("no left subtree for an operator of arity 2")
                self.left = left
                
            
    def __repr__(self):
        return str(self.keyword)
        
    def __str__(self):
        return self.__repr__()

    def print_tree(self, depth):
        print("\t"*depth + str(self))
        if hasattr(self, "left"):
            self.left.print_tree(depth+1)
        if hasattr(self, "right"):
            self.right.print_tree(depth+1)
        
def str2list(query):
    """
    :param query: query as a string with keywords, parenthesis and boolean operators
    :returns: query as a list for processing
    """
    query = query.replace("(", " ( ")
    query = query.replace(")", " ) ")
    query = query.lower()
    query = re.sub("\s+", " ", query)
    if query.startswith(" "):
        query = query[1:]
    if query.endswith(" "):
        query = query[:-1]
    query_list = query.split(" ")
    return query_list
    
def remove_par(query_list):
    res = []
    for w in query_list:
        if w != "(" and w != ")":
            res += [w]
    return res

def get_prios(query_list, prio, k):
    """
    Takes a query as a list of words with parenthesis
    Gives a list of priority order in which to handle the words
    """
    prios = []
    while k < len(query_list):
        w = query_list[k]
        if w == "(":
            subtree, k = get_prios(query_list, prio+4, k+1)
            prios += subtree
        elif w == ")":
            return prios, k+1
        elif w == "and" :
            prios += [prio]
            k+=1
        elif w == "or" :
            prios += [prio+1]
            k+=1
        elif w == "not" :
            prios += [prio+2]
            k+=1
        else:
            prios += [prio+3]
            k+=1
    return prios, k+1

def split2tree(words, prios):
    if len(words) == 1:
        return Node(words[0])
    elif len(words) == 0:
        return None
    else:  
        k = np.argmin(prios)
        return Node(words[k], left=split2tree(words[:k], prios[:k]), right=split2tree(words[k+1:], prios[k+1:]))