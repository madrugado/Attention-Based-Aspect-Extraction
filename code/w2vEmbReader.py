import logging
import os
import re
import numpy as np
import gensim
from sklearn.cluster import KMeans
import pymorphy2

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class W2VEmbReader:
    def __init__(self, data_path, emb_name):
        self.data_path = data_path
        if os.path.sep not in emb_name:
            emb_path = os.path.join(data_path, emb_name)
        else:
            emb_path = emb_name
        logger.info('Loading embeddings from: ' + emb_path)
        self.embeddings = {}
        emb_matrix = []

        model = gensim.models.KeyedVectors.load(emb_path)
        self.emb_dim = model.vector_size
        for word in model.wv.vocab:
            self.embeddings[word] = list(model[word])
            emb_matrix.append(list(model[word]))

        # if emb_dim != None:
        #     assert self.emb_dim == len(self.embeddings['nice'])

        self.vector_size = len(self.embeddings)
        self.emb_matrix = np.asarray(emb_matrix)
        self.aspect_size = None
        logger.info('  #vectors: %i, #dimensions: %i' % (self.vector_size, self.emb_dim))

    def get_emb_given_word(self, word):
        try:
            return self.embeddings[word]
        except KeyError:
            return None

    def get_emb_matrix_given_vocab(self, vocab, emb_matrix):
        counter = 0.
        for word, index in vocab.items():
            try:
                emb_matrix[index] = self.embeddings[word]
                counter += 1
            except KeyError:
                pass

        logger.info(
            '%i/%i word vectors initialized (hit rate: %.2f%%)' % (counter, len(vocab), 100 * counter / len(vocab)))
        # L2 normalization
        norm_emb_matrix = emb_matrix / np.linalg.norm(emb_matrix, axis=-1, keepdims=True)
        return norm_emb_matrix

    def get_aspect_matrix(self, n_clusters=0):
        seed_words_path = os.path.join(self.data_path, "seed_words.txt")
        self.aspect_size = n_clusters
        km = KMeans(n_clusters=n_clusters)
        km.fit(self.emb_matrix)
        km_aspects = km.cluster_centers_
        if os.path.exists(seed_words_path):
            aspects = []
            morph = pymorphy2.MorphAnalyzer()
            with open(seed_words_path) as f:
                for line in f:
                    one_aspect = []
                    for word in re.split('\W+', line.lower()):
                        w = morph.parse(word)[0].normal_form
                        if w in self.embeddings:
                            one_aspect.append(self.embeddings[w])
                    if one_aspect:
                        one_aspect = np.mean(one_aspect, axis=0)
                    else:
                        print("Not initialized:\t" + line)
                        one_aspect = np.random.random((self.emb_dim,))
                    aspects.append(one_aspect)
            self.aspect_size += len(aspects)
            aspects = np.concatenate((km_aspects, np.stack(aspects)), axis=0)
        else:
            aspects = km_aspects
        # L2 normalization
        norm_aspect_matrix = aspects / np.linalg.norm(aspects, axis=-1, keepdims=True)
        return norm_aspect_matrix

    def get_emb_dim(self):
        return self.emb_dim
