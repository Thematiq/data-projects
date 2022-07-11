import numpy as np
import pandas as pd
import pickle
import emoji
import string
import scipy.sparse as sparse
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from enum import Enum
import os

from .params import DIR


# Create your models here.
class Page:
    def __init__(self, title, href, subreddit, short_content, similarity):
        self.title = title
        self.href = href
        self.subreddit = subreddit
        self.short_content = short_content
        self.similarity = similarity


class SearchEngine:
    def __init__(self, db_filename: str, map_filename: str, normal_matrix: str, idf_map: str = None,
                 idf_matrix: str = None, svd: str = None, svd_idf: str = None):
        self.db = pd.read_csv(db_filename)
        with open(map_filename, 'rb') as f:
            self.vocab_map = pickle.load(f)
        if idf_map is None or idf_matrix is None:
            self.idf_map = None
        else:
            with open(idf_map, 'rb') as f:
                self.idf_map = pickle.load(f)
            self.idf_matrix = sparse.load_npz(idf_matrix)
        self.normal_matrix = sparse.load_npz(normal_matrix)
        if svd is None or svd_idf is None:
            self.svd = None
            self.max_k = -1
        else:
            with open(svd_idf, 'rb') as f:
                self.svd_idf = pickle.load(f)
            with open(svd, 'rb') as f:
                self.svd = pickle.load(f)
            self.max_k = min(self.svd[1].shape[0], self.svd_idf[1].shape[0])
            self.svd_cache = {}
            self.svd_idf_cache = {}

    def normal_query(self, q: str, size: int = 10):
        print('Using standard count')
        content = self._preprocess_text(q)
        q = self.veccount_data([content])
        scores = np.asarray((q.T @ self.normal_matrix).todense())[0]
        return self.scores_to_query(scores, size)

    def idf_query(self, q: str, size: int = 10):
        if self.idf_map is None:
            raise AttributeError('Could not find IDF dataset')
        print('Using standard IDF')
        content = self._preprocess_text(q)
        q = self.veccount_data([content], idf_map=self.idf_map)
        scores = np.asarray((q.T @ self.idf_matrix).todense())[0]
        return self.scores_to_query(scores, size)

    def svd_query(self, q: str, size: int = 10, k: int = 100, use_idf: bool = False):
        if self.svd is None:
            raise AttributeError('Could not find SVD dataset')
        if self.max_k < k:
            raise ValueError('K value is larger than maximal allowed')
        if use_idf:
            print('Using SVD IDF')
        else:
            print('Using SVD Count')
        content = self._preprocess_text(q)
        q = self.veccount_data([content])
        if use_idf:
            u, s, v = self.svd_idf
        else:
            u, s, v = self.svd
        scores = ((q.T @ u[:, :k]) @ (s[:k].reshape(-1, 1) * v[:k, :])).flatten()
        if use_idf:
            if k not in self.svd_idf_cache:
                self.svd_idf_cache[k] = np.linalg.norm(s[:k].reshape(-1, 1) * v[:k, :], axis=0)
            scores /= self.svd_idf_cache[k]
        else:
            if k not in self.svd_cache:
                self.svd_cache[k] = np.linalg.norm(s[:k].reshape(-1, 1) * v[:k, :], axis=0)
            scores /= self.svd_cache[k]
        return self.scores_to_query(scores, size)

    def scores_to_query(self, scores, size):
        top = scores.argsort()
        vw = self.db.iloc[top[-size:][::-1]]
        return [
            self._to_page(row, prob)
            for (_, row), prob in zip(vw.iterrows(), scores[top[-size:][::-1]])
        ]

    @staticmethod
    def _preprocess_text(document, drop_numbers=False, drop_punctuations=True, drop_stopwords=True,
                         stem=True, stop_words=None, drop_potential_links=False, transform_emoji=True,
                         drop_nonascii=True):
        if transform_emoji:
            document = emoji.demojize(document)
        if drop_punctuations:
            document = ''.join(char for char in document if char not in string.punctuation)
        if drop_numbers:
            document = ''.join(char for char in document if char not in string.digits)
        if drop_nonascii:
            document = document.encode('ascii', errors='ignore').decode()
        tokens = word_tokenize(document)
        if stem:
            porter = PorterStemmer()
            tokens = [porter.stem(token) for token in tokens]
        if drop_stopwords:
            if stop_words is None:
                stop_words = stopwords.words('english')
            tokens = [token for token in tokens if token not in stop_words]
        if drop_potential_links:
            tokens = list(filter(lambda x: 'http' not in x, tokens))
        return tokens

    def veccount_data(self, data, normalize=True, idf_map=None):
        if idf_map is None:
            idf_map = [1] * len(self.vocab_map)
        matrix = sparse.dok_matrix((len(self.vocab_map), len(data)), dtype='float')
        current_vec = np.empty(len(self.vocab_map))
        for i, doc in enumerate(data):
            current_vec[...] = 0
            # Eval normalization constant
            for member in doc:
                if member in self.vocab_map:
                    current_vec[self.vocab_map[member]] += idf_map[self.vocab_map[member]]
            # Assign elements and normalize them
            vec_norm = np.linalg.norm(current_vec)
            if normalize:
                current_vec /= vec_norm
            for x in np.nonzero(current_vec):
                matrix[x, i] = current_vec[x]
        return matrix.tocsc()

    @staticmethod
    def _to_page(content, prob):
        if pd.isnull(content['body']):
            text = ''
        else:
            text = content['body']
            if len(text) > 100:
                text = text[:100] + '...'
        return Page(
            content['title'],
            content['url'],
            content['subreddit'],
            text,
            '{:.2f}%'.format(100 * prob)
        )


def load_directory(directory: str) -> list:
    REQUIRED_FILES = [
        'smalldb.csv',
        'map.pickle',
        'count.npz'
    ]
    OPTIONAL_FILES = [
        'idf_map.pickle',
        'idf.npz',
        'svd.pickle',
        'svd_idf.pickle'
    ]
    files = list(map(lambda x: os.path.join(directory, x), REQUIRED_FILES))
    if any(map(lambda x: not os.path.exists(x) or not os.path.isfile(x), files)):
        raise IOError('Could not load search engine DB files')
    opt = map(lambda x: os.path.join(directory, x), OPTIONAL_FILES)
    opt = list(map(
        lambda x: x if os.path.exists(x) and os.path.isfile(x) else None,
        opt
    ))
    files.extend(opt)
    return files


static_engine = SearchEngine(*load_directory(DIR))
