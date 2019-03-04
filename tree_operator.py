import numpy as np
import re

class Node:
    """
    Node class for a tree representing a boolean query.
    Each node is an operation, each leaf is a token_id
    """
    
    def __init__(self, keyword, left=None, right=None):
        self.keyword = keyword
        
        if keyword in ["NOT", "AND", "OR"]:
            
            if right is None:
                raise ArgumentError("no right subtree for this operator")
            self.right = right
            if keyword == "OR" or keyword == "AND":
                if left is None:
                    raise ArgumentError("no left subtree for an operator of arity 2")
                self.left = left
                
            
    def __repr__(self):
        return str(self.keyword)
        
    def __str__(self):
        return self.__repr__()

        

def get_tree_rep(ops:list, drawing:str):
    """
    Get a string representing the subtree given by its root.
    Indentation is messy : the first 2 nodes of each line are the left and right 
    """
    line = ""
    line_null = True
    next_line = []
    for op in ops:
        if op is None:
            line += "_\t"
        else:
            line_null = False
            line += str(op) + "\t"
        next_line += [None if not hasattr(op, "left") else op.left,
                      None if not hasattr(op, "right") else op.right]    
    line += "\n"
    if line_null:
        return drawing
    else:
        return get_tree_rep(next_line, drawing+line)

def str2list(query):
    query = query.replace("(", " ( ")
    query = query.replace(")", " ) ")
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
            subtree, k = first_pass_prio(query_list, prio+4, k+1)
            prios += subtree
        elif w == ")":
            return prios, k+1
        elif w == "AND" :
            prios += [prio]
            k+=1
        elif w == "OR" :
            prios += [prio+1]
            k+=1
        elif w == "NOT" :
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


if __name__ == '__main__':
    query = "NOT ((apple AND banana) OR carrot AND donkey) OR (NOT egg)"
    
    print("Converting to list...")
    s = time()
    query_list = str2list(query)
    print("{:.2f}s".format(time()-s))
    print(query_list)
    
    words = remove_par(query_list)
    
    print("First pass...")
    s = time()
    prios, _ = get_prios(query_list, 0, 0)
    print("{:.2f}s".format(time()-s))
    print(prios)
    for i in range(len(words)):
        print("{}\t-\t{}".format(words[i], prios[i]))
    
    print(words)
    tree = split2tree(words, prios)
    
    tree_rep = get_tree_rep([tree], "")
    print("tree rep :")
    print(tree_rep)
    