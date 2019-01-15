import os
from typing import List
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from index import Index
from tree_operator import Operator
from tree_operator import draw

class BooleanIndex(Index):
    def build_cacm(self, filename: str):
        """
        Uses the CACM collection to build the class attributes.
        """
        raw_file = np.loadtxt(filename, dtype=str, delimiter="someneverhappeningstr")
        keep_element = False

        doc_id = -1
        term_id = -1
        title = ''

        nb_terms = 5462
        nb_docs = 3204
        self.incidence_matrix = lil_matrix((nb_terms, nb_docs), dtype=bool)  # Boolean index

        for line in raw_file:
            if line.startswith("."):
                keep_element = False
                is_title = False
                if line.startswith(".I"):
                    if title != '':
                        # Map a doc to its ID
                        self.doc_to_id[str(doc_id + 1) + ' -' + title] = doc_id  # Some titles are not unique
                        self.id_to_doc[doc_id] = str(doc_id + 1) + ' -' + title
                    doc_id += 1  # Real doc ID may be different
                    title = ' '
                elif line.startswith(".T") or line.startswith(".W") or line.startswith(".K"):
                    keep_element = True
                    if line.startswith(".T"):
                        is_title = True

            elif keep_element:
                if is_title:
                    title += line + ' '

                terms = self.normalize(line)

                for term in terms:
                    if term != "":
                        if term not in self.terms_to_id.keys():
                            term_id += 1
                            self.terms_to_id[term] = term_id
                            self.id_to_term[term_id] = term
                        self.incidence_matrix[self.terms_to_id[term], doc_id] = True  # Boolean index

        # Add last doc
        self.doc_to_id[str(doc_id + 1) + ' -' + title] = doc_id
        self.id_to_doc[doc_id] = str(doc_id + 1) + ' -' + title

        self.incidence_matrix = self.incidence_matrix.tocsr()  # Faster row slicing


    def create_bool_tree(self, query: list, left_op:Operator, node_id) -> Operator:  # returns a tree of operations, given by its root
        print("Function called : ", query, left_op, node_id)
        node_id += 1
        my_node_id = node_id
        if len(query) > 0:
            word = query.pop(0)
            if word in ["AND", "NOT", 'OR']:
                if (word == "AND" or word == "OR") and left_op is None:
                    raise ValueError("Query was ill formulated. Please specify a term before an operation \"AND\" or \"OR\"")
                tree_dic[my_node_id] = Operator(keyword=word, left=left_op, right=self.create_bool_tree(query, None, node_id))                

            else:
                if left_op is not None:
                    raise ValueError("Query was ill formulated : Two words without an operator in-between")
                term = self.normalize(word)[0]
                if term != "":
                    token_id = self.terms_to_id[term]  # Get the corresponding id
                    tree_dic[my_node_id] = self.create_bool_tree(query, left_op=Operator(token_id=token_id), node_id=node_id)
                else:
                    raise ValueError("empty word detected") 
            print("{}. Word : {} - Op : {}".format(my_node_id, word, tree_dic[my_node_id]))
            return tree_dic[my_node_id]
        else:
            print("Reached end of query")
            return left_op       
            
    def treat_query(self, query: str) -> Operator:
        print(query)
        query_words = query.split(" ")
        print(query_words)
        return self.create_bool_tree(query_words, None, 0)
          
    def compute_bool_result(self, op):  # recursive function to compute the result of a tree given its root
        if hasattr(op, 'token_id'):
            return self.incidence_matrix[op.token_id]
        else:
            if op.keyword == "OR":
                return self.compute_bool_result(op.left) + self.compute_bool_result(op.right)
            if op.keyword == "AND":
                return self.compute_bool_result(op.left).multiply(self.compute_bool_result(op.right))
            if op.keyword == "NOT":
                return 1 - self.compute_bool_result(op.right)                
        
        
if __name__ == '__main__':
    
    PATH_TO_DATA = 'data'
    print("Creating index...")
    index = BooleanIndex()
    index.build_cacm(os.path.join(PATH_TO_DATA, 'CACM', 'cacm.all'))
    tree_dic = {}
    tree = index.treat_query('Assistant OR program')
    print("word 'program' found in the following docs :")
    print(index.incidence_matrix[14])
    print("____________________________________________")
    print(index.compute_bool_result(tree))
    
    
