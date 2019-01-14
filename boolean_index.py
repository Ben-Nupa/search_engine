import os
from typing import List
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
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

    def build_cs276(self,directory_name : str, from_saved_dict : bool):
        """
        Uses the CS276 (Stanford) collection to build the class attributes
        """
        terms = []
        docs = []
        count = []
        list_of_files = os.listdir(directory_name)
        for fileName in list_of_files:
            file = open(os.path.join(directory_name, fileName), "r")
            tokens_in_file = dict()  # Temporary dictionnary to count the frequency of each token in the document
            content = file.readlines()
            for line in content:
                words = line.split(" ")
                for word in words:
                    if word not in common_words:
                        tokens_in_file[word] = tokens_in_file.get(word, 0) + 1
                        terms.append(term_ids[word])
                        docs.append(doc_ids[fileName])
                        count.append(tokens_in_file[word])
        terms = np.array(terms)
        docs = np.array(docs)
        count = np.array(count)
        self.incidence_matrix = csc_matrix((count, (terms, docs)))

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
    index.build_cacm(os.path.join(PATH_TO_DATA, 'CACM', 'cacm.all'))

    # index.treat_query('Assistant OR program')
