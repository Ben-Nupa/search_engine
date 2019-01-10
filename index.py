import string
from typing import List
from scipy.sparse import lil_matrix
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


class Index:
    """
    Base index class.

    Attributes
    ----------
    terms_to_id: dict
        The vocabulary (stemmed words) of the collection where (key, value) = (term (str), id_term (int)).
    id_to_term: dict
        Same dictionary in reverse order (key, value) = (id_term (int), term (str)).
    doc_to_id: dict
        The collection of documents where (key, value) = ('id_doc - title'(str), id_doc (int)).
    id_to_doc: dict
        Same dictionary in reverse order (key, value) = (id_doc (int), 'id_doc - title'(str)).
    incidence_matrix: dict
        Incidence matrix to map terms to their documents.
    """

    def __init__(self):
        self.terms_to_id = {}
        self.id_to_term = {}
        self.doc_to_id = {}
        self.id_to_doc = {}
        self.incidence_matrix = lil_matrix([])

    def build_cacm(self, filename: str):
        raise NotImplementedError

    def treat_query(self, query: str) -> List[str]:
        raise NotImplementedError

    def compute_similarity(self, query: str):
        raise NotImplementedError

    def build_vector(self, term: str) -> List[bool]:
        raise NotImplementedError

    @staticmethod
    def normalize(sentence: str) -> List[str]:
        """
        Normalize the given sentence by:
            - Transforming the English contractions to full words
            - Transforming composed words into 2 separated words
            - Tokenizing into words based on white space and punctuation
            - Converting to lower case
            - Removing punctuation
            - Removing non-alphabetical tokens
            - Filtering stop words
            - Stemming words

        Inspired from : https://machinelearningmastery.com/clean-text-machine-learning-python/
        """
        # Take care of English contractions, from: https://en.wikipedia.org/wiki/Wikipedia:List_of_English_contractions
        sentence = sentence.replace("'s", ' is')
        sentence = sentence.replace("n't", ' not')
        sentence = sentence.replace("'re", ' are')
        sentence = sentence.replace("'m", ' am')
        sentence = sentence.replace("'ve", ' have')
        sentence = sentence.replace("'ll", ' will')
        sentence = sentence.replace("'d", ' would')
        sentence = sentence.replace("-", ' ')

        # Tokenize
        tokens = word_tokenize(sentence)
        # Convert to lower case
        tokens = [w.lower() for w in tokens]
        # Remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        # Remove remaining tokens that are not alphabetic
        words = [word for word in stripped if word.isalpha()]
        # Filter out stop words
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        # Stemming of words
        porter = PorterStemmer()
        stemmed_words = [porter.stem(word) for word in words]

        return stemmed_words
