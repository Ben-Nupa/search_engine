import os
import sys
from typing import List
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix, load_npz, save_npz, vstack
import pickle

from index import Index
from tree_operator import *

from tqdm import tqdm

from time import time


class BooleanIndex(Index):
    def build_cacm(self, filename: str):
        """
        Uses the CACM collection to build the class attributes.
        """
        raw_file = np.loadtxt(filename, dtype=str, delimiter="\n")
        keep_element = False

        doc_id = -1
        term_id = -1
        title = ''

        nb_terms = 5462 + 2 # for the unknown words (that are in no doc) and stopwords (that are in all docs)
        self.nb_docs = 3204
        self.incidence_matrix = lil_matrix((nb_terms, self.nb_docs), dtype=int)  # Boolean index
        self.terms_to_id["__UNK__"] = nb_terms-2
        self.id_to_term[nb_terms-2] = "__UNK__"
        self.terms_to_id["__stopword__"] = nb_terms-1
        self.id_to_term[nb_terms-1] = "__stopword__"
        for i in range(self.nb_docs):
            self.incidence_matrix[nb_terms-1, i] = 1
        
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

    def build_cs276(self, directory_name: str):
        """
        Uses the CS276 (Stanford) collection to build boolean_index and saves

        Args :
            directory_name : String containing the path to the CS276 dataset (pa1-data)
        """

        # Creating directory if not existing :
        if not os.path.exists(os.path.join("CS276_boolean_index")):
            os.makedirs(os.path.join("CS276_boolean_index"))

        doc_id = -1
        term_id = -1

        nb_terms = 353975 +2
        self.nb_docs = 98998

        for block_id in range(10):
            list_of_files = os.listdir(os.path.join(directory_name, str(block_id)))
            block_inc_matrix = lil_matrix((nb_terms, self.nb_docs))
            print("Building index for block {}:".format(block_id))
            for file_id in tqdm(range(len(list_of_files))):
                # Reading the document
                file = open(os.path.join(directory_name, str(block_id), list_of_files[file_id]), "r")
                content = file.readlines()
                file.close()
                # Adding the document to both doc_to_id and id_to_doc dictionaries
                doc_id += 1
                doc = os.path.join(str(block_id), list_of_files[file_id])
                self.doc_to_id[doc] = doc_id
                self.id_to_doc[doc_id] = doc
                # Counting the terms
                for line in content:
                    terms = self.normalize(line)
                    for term in terms:
                        if term != "":
                            if term not in self.terms_to_id.keys():
                                term_id += 1
                                self.terms_to_id[term] = term_id
                                self.id_to_term[term_id] = term
                            block_inc_matrix[
                                self.terms_to_id[term], doc_id] = 1  # +=1 if we want to count the frequency of the term
            print("Saving block " + str(block_id))
            block_inc_matrix = block_inc_matrix.tocsc()
            save_npz(os.path.join("CS276_boolean_index","block_inc_matrix" + str(block_id) + ".npz"), block_inc_matrix)

        #Saving the four dictionaries
        print("Saving dictionaries")
        with open(os.path.join("CS276_boolean_index",'doc_to_id.pkl'), 'wb') as output_doc_to_id :
            pickle.dump(self.doc_to_id, output_doc_to_id)
        with open(os.path.join("CS276_boolean_index",'id_to_doc.pkl'), 'wb') as output_id_to_doc :
            pickle.dump(self.id_to_doc, output_id_to_doc)
        with open(os.path.join("CS276_boolean_index",'id_to_term.pkl'), 'wb') as output_id_to_term :
            pickle.dump(self.id_to_term, output_id_to_term)
        with open(os.path.join("CS276_boolean_index",'term_to_id.pkl'), 'wb') as output_term_to_id :
            pickle.dump(self.terms_to_id, output_term_to_id)

    def load_cs276_index(self):
        """
        Load the CS276 saved index in .npz format (the build_cs276 function should have been run before
        in order to have the .npz files)
        """
        nb_terms = 353975
        self.nb_docs = 98998
        print("Loading index matrices")
        self.incidence_matrix = csc_matrix((nb_terms, self.nb_docs))
        for block_id in range(10):
            self.incidence_matrix += load_npz(os.path.join("CS276_boolean_index","block_inc_matrix" + str(block_id) + ".npz"))
        self.incidence_matrix = self.incidence_matrix.tocsr()
        #Loading the four dictionaries
        print("Loading dictionaries")
        with open(os.path.join("CS276_boolean_index",'doc_to_id.pkl'), 'rb') as input_doc_to_id :
            self.doc_to_id = pickle.load(input_doc_to_id)
        with open(os.path.join("CS276_boolean_index",'id_to_doc.pkl'), 'rb') as input_id_to_doc :
            self.id_to_doc = pickle.load(input_id_to_doc)
        with open(os.path.join("CS276_boolean_index",'id_to_term.pkl'), 'rb') as input_id_to_term :
            self.id_to_term = pickle.load(input_id_to_term)
        with open(os.path.join("CS276_boolean_index",'term_to_id.pkl'), 'rb') as input_term_to_id :
            self.terms_to_id = pickle.load(input_term_to_id)



    def compute_bool_result(self, op):  
        """
        Recursive function to compute the result of a tree given its root
        
        Returns a list of postings and a boolean indicating whether 
        it is a positive list (the result is the list) 
        or a negative list (the result is all docs EXCEPT those in the list)
        """
        if op.keyword == "or":
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
             
        elif op.keyword == "and":
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
                            
        elif op.keyword == "not":
            right_result, right_pos = self.compute_bool_result(op.right)        
            return right_result, not right_pos
        else:
            if op.keyword in self.terms_to_id:
                row = self.incidence_matrix[self.terms_to_id[op.keyword]]
            else:  # __UNK__ and __stopword__ should be in the dic. If not, we are prepared
                row = lil_matrix((1, self.nb_docs), dtype=int).tocsr()
                if op.keyword == "__stopword__":
                    for i in range(self.nb_docs):
                        row[0, i] = 1
            return row, True


    def treat_query(self, query: str, show_tree: bool=False):
        
        query_list = str2list(query)
        query_words = remove_par(query_list)
        for i in range(len(query_words)):
            if query_words[i] not in ["and", "or", "not"]:
                normalized_word = self.normalize(query_words[i])  # is a list of 1 element
                if len(normalized_word) == 0: # word was a stop word and got filtered
                    print("Warning : word \"{}\" is actually a stopword".format(query_words[i]))
                    query_words[i] = "__stopword__"
                elif normalized_word[0] not in self.terms_to_id:
                    print("Warning : word \"{}\" not in the index".format(query_words[i]))
                    query_words[i] = "__UNK__"
                else:
                    query_words[i] = normalized_word[0]
                    
        prios, _ = get_prios(query_list, 0, 0)
        tree = split2tree(query_words, prios)
        if show_tree:
            print(query_words)
            tree.print_tree(0)
            
        result, pos_list = self.compute_bool_result(tree)
        
        if pos_list:
            _, doc_ids = result.nonzero()
        else:
            ones = np.ones(result.shape, dtype=int)
            _, doc_ids = (ones - result).nonzero()
            
        docs = np.array([doc_id + 1 for doc_id in doc_ids])  # doc ids in the database start at 1

        return docs
    
    
if __name__ == '__main__':
    PATH_TO_DATA = 'data'
    print("Creating index...")
    start = time()
    index = BooleanIndex()
    print("Index declared... {:.2f}s".format(time()-start))
    start = time()
    #index.build_cacm(os.path.join(PATH_TO_DATA, 'CACM', 'cacm.all'))
    index.load_cs276_index()
    print("Index built ! {:.2f}s".format(time()-start))

    result = index.treat_query('Assistant OR program')
    result = index.treat_query('Assistant OR (program AND NOT tendency AND minimum AND successful)')
    print("result : {}".format(result))
