from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import codecs
import json
from tqdm import tqdm
import argparse


def parseSentence(line):
    lmtzr = WordNetLemmatizer()
    stop = stopwords.words('english')
    text_token = CountVectorizer().build_tokenizer()(line.lower())
    text_rmstop = [i for i in text_token if i not in stop]
    text_stem = [lmtzr.lemmatize(w) for w in text_rmstop]
    return text_stem


def preprocess_train(domain):
    f = codecs.open('../datasets/' + domain + '/train.txt', 'r', 'utf-8')
    out = codecs.open('../preprocessed_data/' + domain + '/train.txt', 'w', 'utf-8')

    for line in f:
        tokens = parseSentence(line)
        if len(tokens) > 0:
            out.write(' '.join(tokens) + '\n')


def preprocess_test(domain):
    # For restaurant domain, only keep sentences with single 
    # aspect label that in {Food, Staff, Ambience}

    f1 = codecs.open('../datasets/' + domain + '/test.txt', 'r', 'utf-8')
    f2 = codecs.open('../datasets/' + domain + '/test_label.txt', 'r', 'utf-8')
    out1 = codecs.open('../preprocessed_data/' + domain + '/test.txt', 'w', 'utf-8')
    out2 = codecs.open('../preprocessed_data/' + domain + '/test_label.txt', 'w', 'utf-8')

    for text, label in zip(f1, f2):
        label = label.strip()
        if domain == 'restaurant' and label not in ['Food', 'Staff', 'Ambience']:
            continue
        tokens = parseSentence(text)
        if len(tokens) > 0:
            out1.write(' '.join(tokens) + '\n')
            out2.write(label + '\n')


def preprocess_line(line):
    return " ".join([morph.parse(w)[0].normal_form for w in word_tokenize(line.lower())])


def preprocess_reviews_train():
    with open("../preprocessed_data/app_reviews/appstore.json", "rt") as f:
        reviews = json.load(f)
    with open("../preprocessed_data/app_reviews/train.txt", "wt") as f:
        for rev in tqdm(reviews):
            if isinstance(rev, dict):
                f.write(preprocess_line(rev["Title"] + " " + rev["Review"]) + "\n")


def preprocess(domain):
    print('\t' + domain + ' train set ...')
    preprocess_train(domain)
    print('\t' + domain + ' test set ...')
    preprocess_test(domain)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", dest="domain", type=str, metavar='<str>', default='restaurant',
                        help="domain of the corpus")
    args = parser.parse_args()

    if args.domain == "app_reviews":
        import pymorphy2
        from nltk.tokenize import word_tokenize

        morph = pymorphy2.MorphAnalyzer()

        print('Preprocessing raw review sentences ...')
        preprocess_reviews_train()
    else:
        preprocess(args.domain)
