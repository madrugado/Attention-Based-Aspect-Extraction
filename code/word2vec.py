import gensim
import codecs
import argparse


class MySentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in codecs.open(self.filename, 'r', 'utf-8'):
            yield line.split()


def word2vec(domain):
    print('Pre-training word embeddings for %s ...' % (domain))
    source = '../preprocessed_data/%s/train.txt' % (domain)
    model_file = '../preprocessed_data/%s/w2v_embedding' % (domain)
    sentences = MySentences(source)
    model = gensim.models.Word2Vec(sentences, size=200, window=10, min_count=5, workers=4)
    model.save(model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", dest="domain", type=str, metavar='<str>', default='restaurant',
                        help="domain of the corpus")
    args = parser.parse_args()

    word2vec(args.domain)
