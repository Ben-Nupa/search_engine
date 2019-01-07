import os
from typing import List
import numpy as np
from scipy.sparse import lil_matrix


class Index:
    """
    Base index class.

    Attributes
    ----------
    terms: dict
        The vocabulary of the collection where (key, value) = (term (str), id_term (int)).
    documents: dict
        The collection of documents where (key, value) = (id doc (int), content of doc (list)).
    """

    def __init__(self):
        self.terms_to_id = {}
        self.id_to_term = {}
        # self.documents = {}
        self.doc_to_id = {}
        self.id_to_doc = {}
        # self.inverted_index = [] # list of numpy arrays
        self.incidence_matrix = lil_matrix([])  # Unsure if it's most efficient

    def build(self, filename: str):
        raise NotImplementedError

    def treat_query(self, query: str) -> List[str]:
        raise NotImplementedError

    def compute_similarity(self, query: str):
        raise NotImplementedError

    def build_vector(self, term: str) -> List[bool]:
        raise NotImplementedError

    @staticmethod
    def normalize(term: str) -> str:
        term = term.lower()
        term.replace("; ", " ")
        term.replace(", ", " ")
        term.replace("(", "")
        term.replace(")", "")
        return term


class BooleanIndex(Index):
    def __init__(self, filename: str):
        super().__init__()
        # self.build(filename)

    # TODO : transform into a matrix/sparse matrix ?
    def build_cacm(self, filename: str):
        # filename = 'cacm.all'
        raw_file = np.loadtxt(filename, dtype=str, delimiter="someneverhappeningstr")
        keep_element = False

        doc_id = -1
        term_id = -1

        nb_terms = 17542
        nb_docs = 3204  # nb of docs in cacm.all
        self.incidence_matrix = lil_matrix((nb_terms, nb_docs))

        for row in raw_file:
            if row.startswith("."):
                if row.startswith(".I"):
                    doc_id += 1  # assumption : docs are read in order
                    # if nb_docs / 2 <= doc_id < nb_docs / 2 + 1:
                    #     print("At half the docs : ")
                    #     print("Vocabulary size : {}".format(len(term_dic)))
                    #     print("Nb tokens : {}".format(sum(term_dic.values())))
                elif row.startswith(".T") or row.startswith(".W") or row.startswith(".K"):
                    keep_element = True
                    if row.startswith(".T"):
                        was_title = True
                    else:
                        was_title = False
                else:
                    keep_element = False
            elif keep_element:
                # Map a doc to is ID
                if was_title:
                    self.doc_to_id[row + str(doc_id)] = doc_id  # Some titles are not unique
                    self.id_to_doc[doc_id] = row + str(doc_id)
                    was_title = False

                row = self.normalize(row)

                terms = row.split(" ")
                for term in terms:
                    # term = self.normalize(term)  # TODO
                    if term != "":
                        if term not in self.terms_to_id.keys():
                            term_id += 1
                            self.terms_to_id[term] = term_id
                            self.id_to_term[term_id] = term
                        # if term_id > 17541:
                        #     print(term_id)
                        # if doc_id > 3200:
                        #     print(doc_id)
                        self.incidence_matrix[term_id, doc_id] = 1  # Boolean index

        self.incidence_matrix = self.incidence_matrix.tocsc()  # Faster column slicing
        print(len(self.doc_to_id), len(self.id_to_doc), len(self.terms_to_id), len(self.id_to_term))
        print(self.incidence_matrix.shape)
        print(doc_id)

    def treat_query(self, query: str) -> np.array:
        query_words = query.split()
        bool_operations = []
        bool_terms = []
        for word in query_words:
            if word in ["AND", "NOT", 'OR']:  # TODO : add OR
                bool_operations.append(word)
            else:
                term = self.normalize(word)
                # TODO : check if term != ''
                bool_terms.append(self.terms_to_id[term])  # Get the corresponding id
        bool_result = self.compute_bool_result(bool_operations, bool_terms)
        return np.where(bool_result == 1)[0]

    def compute_bool_result(self, operations: List[str], bool_terms: List[int]) -> lil_matrix:
        result = self.incidence_matrix[bool_terms[0], :]
        idx_term = 1
        for i in range(len(operations)):
            term = []
            if operations[i] == "NOT":
                term = [abs(coefficient - 1) for coefficient in bool_terms[i + 1]]  # TODO: adapt to sparse
            elif operations[i] == "AND":
                term = self.incidence_matrix[bool_terms[0], :]
            result = result.dot(term)
        return result


if __name__ == '__main__':
    PATH_TO_DATA = 'data'
    index = BooleanIndex('')
    index.build_cacm(os.path.join(PATH_TO_DATA, 'CACM', 'cacm.all'))
