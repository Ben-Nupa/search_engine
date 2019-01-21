import os
import time
from typing import List
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix, csr_matrix
from scipy.sparse.linalg import norm as sparse_norm
from index import Index


class VectorizedIndex(Index):
    def __init__(self):
        super().__init__()
        self.idf_vector = np.array([])

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
                        self.incidence_matrix[self.terms_to_id[term], doc_id] += 1  # TF

        self.incidence_matrix = self.incidence_matrix.tocsc()  # Faster column slicing

        # Add last doc
        self.doc_to_id[str(doc_id + 1) + ' -' + title] = doc_id
        self.id_to_doc[doc_id] = str(doc_id + 1) + ' -' + title

        # Compute and store IDF (for queries)
        self.idf_vector = np.zeros((nb_terms, 1))
        for i, row in enumerate(self.incidence_matrix):
            nb_doc_with_term = len(row.nonzero()[0])
            self.idf_vector[i, 0] = np.log(nb_docs / nb_doc_with_term)

        # Add IDF to incidence matrix and normalize it
        computation_version = 3  # 3 is faster
        if computation_version == 1:  # for loop
            for j in range(nb_docs):
                self.incidence_matrix[:, j] = self.incidence_matrix[:, j].multiply(self.idf_vector)
        elif computation_version == 2:  # double inverse
            self.incidence_matrix = csc_matrix(self.incidence_matrix / (1 / self.idf_vector))
        elif computation_version == 3:  # using whole matrix
            temp_matrix = np.ones(self.incidence_matrix.shape) * self.idf_vector
            self.incidence_matrix = self.incidence_matrix.multiply(temp_matrix)

        self.idf_vector = self.idf_vector.T  # Useful for query treat

        # Normalize
        self.incidence_matrix /= sparse_norm(self.incidence_matrix, axis=0)

        # Add last doc
        self.doc_to_id[str(doc_id + 1) + ' -' + title] = doc_id
        self.id_to_doc[doc_id] = str(doc_id + 1) + ' -' + title

    def _build_tf_idf_vector(self, terms: list) -> csc_matrix:
        vector = lil_matrix((1, self.incidence_matrix.shape[0]))
        # TF
        for term in terms:
            if term in self.terms_to_id.keys():
                vector[0, self.terms_to_id[term]] += 1
        # IDF
        vector = vector.tocsc().multiply(self.idf_vector)
        # Normalize
        vector /= sparse_norm(vector)
        return vector

    def treat_query(self, query: str):
        query_terms = self.normalize(query)
        query_vector = self._build_tf_idf_vector(query_terms)
        cosine_vector = self.cosine_similarity(query_vector)  # TODO: change to compute_similarity
        # Order the results by descending order of TF-IDF
        order = (-cosine_vector).argsort() + 1  # Document ID begins at 1
        return order

    def cosine_similarity(self, vector: csc_matrix) -> csc_matrix:
        return vector.dot(self.incidence_matrix)


if __name__ == '__main__':
    PATH_TO_DATA = 'data'

    begin = time.time()

    index = VectorizedIndex()
    index.build_cacm(os.path.join(PATH_TO_DATA, 'CACM', 'cacm.all'))
    print('Construction time = ', time.time() - begin)
    print('######################')

    begin = time.time()
    print(index.treat_query(
        'The General Recursive Algebra and Differentiation Assistant (GRAD Assistant) now under development is a set of LISP functions which symbolically manipulate abd differentiate algebraic expressions. It is designed for use with problemms in which a large amount of routine manipulation is to be done by a program without human intervention. Thus, GRAD must recognize necessary simplifications without external guidance. While some complicated expressions (notably ones involving nested radicals and trigonometric functions) do not yield completely to the present version, it has proved quite useful indeed.'))
    print('Query time = ', time.time() - begin)
