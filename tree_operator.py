import numpy as np

class Operator:
    """
    Node class for a tree representing a boolean query.
    Each node is an operation, each leaf is a token_id
    """
    
    def __init__(self, keyword=None, left=None, right=None, token_id=None):
        
        if keyword is not None:
            self.keyword = keyword  # operation ("OR", "AND", "NOT")
            
            if right is None:
                raise ArgumentError("no right subtree for this operator")
            self.right = right
            if keyword == "OR" or keyword == "AND":
                if left is None:
                    raise ArgumentError("no left subtree for an operator of arity 2")
                self.left = left
                
        else:
            if token_id is None:
                raise ArgumentError("Operator has been specified neither as a node or a leaf")
            self.token_id=token_id
            
    def __repr__(self):
        if hasattr(self, "keyword"):
            s = str(self.keyword)
        elif hasattr(self, "token_id"):
            s = str(self.token_id)
        return s
        
    def __str__(self):
        return self.__repr__()
        
        
def draw(ops:list, drawing:list):
    line = ""
    line_null = True
    next_line = []
    for op in ops:
        if op == "[]":
            line += "[]\t"
        else:
            line_null = False
            line += str(op) + "\t"
        next_line += ["[]" if not hasattr(op, "left") else op.left,
                      "[]" if not hasattr(op, "right") else op.right]
    #line += "\n"
    
    if line_null:
        return drawing
    else:
        # for i in range(len(drawing)):
        #     drawing[i] = "\t" + drawing[i]

        return draw(next_line, drawing+[line])