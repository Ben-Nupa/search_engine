import os
import sys
from typing import List
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from index import Index
from tree_operator import Node
from tree_operator import get_prios
from tree_operator import split2tree
from tree_operator import get_tree_rep
from tree_operator import str2list
from tree_operator import remove_par

from time import time


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
        self.incidence_matrix = lil_matrix((nb_terms, nb_docs), dtype=int)  # Boolean index

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
                        self.incidence_matrix[self.terms_to_id[term], doc_id] = 1  # Boolean index

        # Add last doc
        self.doc_to_id[str(doc_id + 1) + ' -' + title] = doc_id
        self.id_to_doc[doc_id] = str(doc_id + 1) + ' -' + title

        self.incidence_matrix = self.incidence_matrix.tocsr()  # Faster row slicing

        
    def compute_bool_result(self, op):  
        """
        Recursive function to compute the result of a tree given its root
        
        Returns a list of postings and a boolean indicating whether 
        it is a positive list (the result is the list) 
        or a negative list (the result is all docs EXCEPT those in the list)
        """
        if op.keyword == "OR":
            left_result, left_pos = self.compute_bool_result(op.left)
            right_result, right_pos = self.compute_bool_result(op.right)
            if left_pos and right_pos:
                return (left_result + right_result).sign(), True
            elif not left_pos and not right_pos:
                return left_result.multiply(right_result), False
            elif left_pos and not right_pos:
                return (right_result-left_result > 0 ).astype(int), False  # you are encouraged to check yourself this formula works
            elif not left_pos and right_pos:
                return (left_result-right_result > 0 ).astype(int), False  # you are encouraged to check yourself this formula works           
             
        elif op.keyword == "AND":
            left_result, left_pos = self.compute_bool_result(op.left)
            right_result, right_pos = self.compute_bool_result(op.right)
            if left_pos and right_pos:
                return left_result.multiply(right_result), True
            elif not left_pos and not right_pos:
                return (left_result + right_result).sign(), False
            elif left_pos and not right_pos:
                return (left_result-right_result> 0 ).astype(int), False  # you are encouraged to check yourself this formula works
            elif not left_pos and right_pos:
                return (right_result-left_result > 0 ).astype(int), False # you are encouraged to check yourself this formula works           
                            
        elif op.keyword == "NOT":
            right_result, right_pos = self.compute_bool_result(op.right)        
            return right_result, not right_pos
            
        else:
            return self.incidence_matrix[self.terms_to_id[op.keyword]], True
    
    def treat_query(self, query: str):
        
        query_list = str2list(query)
        query_words = remove_par(query_list)
        for i in range(len(query_words)):
            if query_words[i] not in ["AND", "OR", "NOT"]:
                query_words[i] = self.normalize(query_words[i])[0]
        print("query : {}".format(query_list))
        print("Creating query tree...")
        start = time()
        
        prios, _ = get_prios(query_list, 0, 0)
        tree = split2tree(query_words, prios)

        print("Tree declared ! {:.2f} s".format(time()-start))
        print(get_tree_rep([tree], ""))

        print("Computing query result...")
        start = time()
        result, pos_list = self.compute_bool_result(tree)
        print("Done ! {:.2f}s".format(time()-start))
        
        if pos_list:
            return result
        else:
            ones = np.ones(result.shape, dtype=int)
            return ones - result
    
    
    def input_query(self):
        query = input("Enter a boolean query : \n(Example : NOT Assistant OR program AND paper)")
        return self.treat_query(query)    
    
    
if __name__ == '__main__':
    PATH_TO_DATA = 'data'
    print("Creating index...")
    start = time()
    index = BooleanIndex()
    print("Index declared... {:.2f}s".format(time()-start))
    start = time()
    index.build_cacm(os.path.join(PATH_TO_DATA, 'CACM', 'cacm.all'))
    print("Index built ! {:.2f}s".format(time()-start))

    result = index.treat_query('Assistant OR program')
    result = index.treat_query('Assistant OR program AND NOT tendency AND minimum AND successful')
    print("result : {}".format(result))
