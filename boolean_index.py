import os
from typing import List
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix, save_npz
from index import Index


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
        self.incidence_matrix = lil_matrix((nb_terms, nb_docs))

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
                        self.incidence_matrix[term_id, doc_id] = 1  # Boolean index

        # Add last doc
        self.doc_to_id[str(doc_id + 1) + ' -' + title] = doc_id
        self.id_to_doc[doc_id] = str(doc_id + 1) + ' -' + title

        self.incidence_matrix = self.incidence_matrix.tocsc()  # Faster column slicing

    def build_cs276(self,directory_name : str, from_saved_matrix : bool):
        """
        Uses the CS276 (Stanford) collection to build the class attributes

        Args :
            directory_name : String containing the path to the CS276 dataset (pa1-data)
            from_saved_matrix  : True in order not to read again the whole dataset and load a saved matrix
        """
        doc_id = -1
        term_id = -1

        nb_terms = 353975
        nb_docs = 98998
        self.incidence_matrix = lil_matrix((nb_terms, nb_docs))

        for block_id in range(10) :
            list_of_files = os.listdir(os.path.join(directory_name, str(block_id)))
            block_inc_matrix = lil_matrix((nb_terms,nb_docs))
            for filename in list_of_files:
                file = open(os.path.join(directory_name, str(block_id), filename), "r")
                # Adding the document to both doc_to_id and id_to_doc dictionaries
                doc_id += 1
                doc = os.path.join(str(block_id),filename)
                self.doc_to_id[doc] = doc_id
                self.id_to_doc[doc_id] = doc
                #Reading the document
                content = file.readlines()
                file.close()
                for line in content:
                    terms = self.normalize(line)
                    for term in terms:
                        if term != "":
                            if term not in self.terms_to_id.keys():
                                term_id += 1
                                self.terms_to_id[term] = term_id
                                self.id_to_term[term_id] = term
                            #self.incidence_matrix[self.terms_to_id[term], doc_id] = 1 # +=1 if we want to count the frequency of the term
                            block_inc_matrix[self.terms_to_id[term], doc_id] = 1
            print("Saving block " + str(block_id))
            save_npz("block_inc_matrix"+str(block_id)+".npz", block_inc_matrix)
        #self.incidence_matrix = self.incidence_matrix.tocsc()  # Faster column slicing
        #save_npz("incidence_matrix.npz", self.incidence_matrix)

    def treat_query(self, query: str) -> np.array:
        print(query)
        query_words = query.split()
        bool_operations = []
        bool_terms = []
        for word in query_words:
            if word in ["AND", "NOT", 'OR']:
                bool_operations.append(word)
            else:
                term = self.normalize(word)
                # TODO : check if term != ''
                bool_terms.append(self.terms_to_id[term])  # Get the corresponding id
        bool_result = self.compute_bool_result(bool_operations, bool_terms)

        for idx_doc in bool_result.nonzero()[1]:
            print(self.id_to_doc[idx_doc])
        return bool_result.nonzero()

    def compute_bool_result(self, operations: List[str], bool_terms: List[int]) -> lil_matrix:
        result = self.incidence_matrix[bool_terms[0], :]
        for i in range(len(operations)):
            if operations[i] == 'NOT':
                term = np.absolute(1 - self.incidence_matrix[bool_terms[i + 1], :].toarray())
                result = result.multiply(csc_matrix(term))
            elif operations[i] == 'AND':
                result = result.multiply(self.incidence_matrix[bool_terms[i + 1], :])
            elif operations[i] == 'OR':
                result = result.maximum(self.incidence_matrix[bool_terms[i + 1], :])
        return result


if __name__ == '__main__':
    PATH_TO_DATA = 'data'

    index = BooleanIndex()
    #index.build_cacm(os.path.join(PATH_TO_DATA, 'CACM', 'cacm.all'))
    index.build_cs276(os.path.join("..", "pa1-data", "pa1-data"),False)
    # index.treat_query('Assistant OR program')
